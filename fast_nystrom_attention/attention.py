"""
FastNystromAttention - A drop-in replacement for PyTorch's MultiheadAttention
that uses the Nyström method to approximate attention with better efficiency.
"""
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from einops import rearrange
from torch_cluster import fps



def sample_landmarks(
    x: torch.Tensor, 
    num_landmarks: int, 
    sample_method: str = "fps",
    guarantee_mask: Optional[torch.Tensor] = None,
    exclude_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Sample landmarks from input points using specified method.
    
    Args:
        x: Input tensor of shape [bsz, N, D] or [bsz, H, N, D]
        num_landmarks: Number of landmarks to sample
        sample_method: Sampling method ("fps" or "random")
        guarantee_mask: Boolean mask of points that must be included [bsz, N]
        exclude_mask: Boolean mask of points that cannot be included [bsz, N]
    
    Returns:
        Indices of sampled landmarks [bsz, num_landmarks]
    """

    device = x.device
    bsz = x.shape[0]
    N = x.shape[-2]

    # Initialize masks
    if guarantee_mask is None:
        guarantee_mask = torch.zeros(bsz, N, dtype=torch.bool, device=device)
    if exclude_mask is None:
        exclude_mask = torch.zeros(bsz, N, dtype=torch.bool, device=device)

    # Masks
    restricted_mask = ~guarantee_mask & ~exclude_mask           # eligible for sampling
    available_mask = ~exclude_mask                              # all usable points
    guarantee_count = guarantee_mask.sum(dim=1)                 # [bsz]
    available_count = available_mask.sum(dim=1)                 # [bsz]
    restricted_samples = num_landmarks - guarantee_count        # [bsz]
    max_restricted = restricted_samples.max().item()

    # Handle full fallback case where num_landmarks >= available_count
    fallback = num_landmarks >= available_count                 # [bsz]
    all_indices = torch.arange(N, device=device).unsqueeze(0).expand(bsz, -1)  # [bsz, N]

    valid_indices = torch.where(available_mask, all_indices, N)  # pad invalids with N (out-of-range sentinel)
    valid_indices = valid_indices.sort(dim=1).values[:, :num_landmarks]       # truncate to num_landmarks
    valid_indices[fallback.unsqueeze(1).expand_as(valid_indices) == 0] = N     # mask non-fallback rows

    # Top-k restricted indices for sampling
    restricted_count = restricted_mask.sum(dim=1)
    max_count = restricted_count.max().item()
    topk_indices = torch.topk(restricted_mask.int(), k=max_count, dim=1).indices  # [bsz, max_count]

    # Flatten x for sampling
    if x.ndim == 4:
        points = rearrange(x, 'b h n d -> b n (h d)')
    elif x.ndim == 3:
        points = x
    else:
        raise ValueError(f"Invalid input shape: {x.shape}")

    points_for_sampling = points.gather(1, topk_indices.unsqueeze(-1).expand(-1, -1, points.shape[-1]))

    if sample_method == "fps":
        flat = rearrange(points_for_sampling, 'b n d -> (b n) d')
        batch = torch.arange(bsz, device=device).repeat_interleave(max_count)
        ratio = restricted_samples.float() / restricted_count.clamp(min=1).float()
        fps_indices = fps(flat, batch, ratio=ratio.max().item())  # (global indices)
        batch_ids = batch[fps_indices]
        local = fps_indices - batch_ids * max_count
        fps_local = local.reshape(bsz, -1)[:, :max_restricted]
        sampled = topk_indices.gather(1, fps_local)
    elif sample_method == "random":
        rand = torch.rand(bsz, max_count, device=device)
        perm = rand.argsort(dim=1)
        shuffled = topk_indices.gather(1, perm)
        sampled = shuffled[:, :max_restricted]
    else:
        raise ValueError(f"Unknown method: {sample_method}")

    # Guaranteed indices
    guarantee_indices = torch.where(guarantee_mask, all_indices, N)
    guarantee_indices = guarantee_indices.sort(dim=1).values
    max_guarantee = guarantee_count.max().item()
    guarantee_indices = guarantee_indices[:, :max_guarantee]

    # Pad sampled to ensure combined has num_landmarks
    sampled_pad_len = (num_landmarks - max_guarantee) - sampled.shape[1]
    if sampled_pad_len > 0:
        sampled_pad = torch.full((bsz, sampled_pad_len), -1, dtype=torch.long, device=device)
        sampled = torch.cat([sampled, sampled_pad], dim=1)

    # Combine sampled + guaranteed
    combined = torch.cat([sampled, guarantee_indices], dim=1)[:, :num_landmarks]

    # Pad fallback indices to shape
    pad_len = num_landmarks - valid_indices.shape[1]
    if pad_len > 0:
        pad = torch.full((bsz, pad_len), -1, dtype=torch.long, device=device)
        valid_indices = torch.cat([valid_indices, pad], dim=1)
    else:
        valid_indices = valid_indices[:, :num_landmarks]

    # Final: use fallback for batches that need it
    return torch.where(fallback.unsqueeze(1), valid_indices, combined)


def _invert(A: torch.Tensor) -> torch.Tensor:
    """
    Efficiently invert a matrix using Newton-Schulz iteration.
    
    This is the exact coefficient computation, 1 / ||K||_1, of initialization of Z_0, 
    leading to faster convergence.
    
    Args:
        A: A square matrix to invert
        
    Returns:
        The inverted matrix
    """
    # Create identity matrix with same dtype and device as input
    I = torch.eye(A.shape[-1], device=A.device, dtype=A.dtype)
    Z = 1 / torch.max(torch.sum(A, dim=-2, keepdim=True), dim=-1, keepdim=True).values * A.mT
    for _ in range(6):
        AZ = A @ Z
        Z = 0.25 * Z @ (13 * I - AZ @ (15 * I - AZ @ (7 * I - AZ)))
    return Z


def fast_nystrom_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    sample_indices: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
    return_kv_landmarks: bool = False
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Functional implementation of Fast Nyström Attention.
    
    Args:
        query: Query tensor
        key: Key tensor
        value: Value tensor
        sample_indices: Indices of landmark points for Nyström approximation
        attn_mask: Mask to prevent attention to certain positions
        is_causal: Whether to apply causal mask
        
    Returns:
        - Output tensor
    """
    # For self-attention, key and value are the same as query
    #assert key is query and value is query, "Current implementation only supports self-attention"
    
    # Get device
    device = query.device
    
    # Get dimensions
    bsz = query.shape[0]
    bsz_index = torch.arange(bsz, device=device)[:, None]
    head_dim = query.shape[-1]
    scale = head_dim ** -0.5 if scale is None else scale
    
    def index(t: torch.Tensor, sample_indices: torch.Tensor) -> torch.Tensor:
        """Helper function to index tensors for landmark selection."""
        return rearrange(t[bsz_index, :, sample_indices, :], "bsz s h d -> bsz h s d")

    # Extract landmark points
    qp, kp = index(query, sample_indices), index(key, sample_indices)
    
    # Compute Nyström approximation components
    Bk = torch.softmax(query @ (scale * kp.mT), dim=-1)
    Bv = F.scaled_dot_product_attention(qp, key, value)
    A = rearrange(Bk[bsz_index, :, sample_indices, :], "bsz s1 h s2 -> bsz h s1 s2")
    
    # Solve the system using the Nyström method
    vp = _invert(A) @ Bv
    x = Bk @ vp
    
    if return_kv_landmarks:
        return x, (kp, vp)
    return x


class FastNystromAttention(nn.MultiheadAttention):
    """
    Fast Nyström Attention as a drop-in replacement for PyTorch's MultiheadAttention.
    
    This implementation approximates the attention matrix using the Nyström method,
    which reduces computational complexity for long sequences from O(N²) to O(N),
    where N is the sequence length.
    
    Args:
        embed_dim (int): The embedding dimension
        num_heads (int): Number of attention heads
        dropout (float, optional): Dropout probability. Default: 0.0
        bias (bool, optional): If True, adds bias to the projections. Default: True
        add_bias_kv (bool, optional): If True, adds bias to the key and value projections. Default: False
        add_zero_attn (bool, optional): If True, adds a new batch of zeros to the key and value. Default: False
        kdim (int, optional): Dimension of the key. Default: None (=embed_dim)
        vdim (int, optional): Dimension of the value. Default: None (=embed_dim)
        batch_first (bool, optional): If True, input and output tensors are provided as (batch, seq, feature). Default: False
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
        batch_first: bool = False,
    ) -> None:
        super().__init__(
            embed_dim, num_heads, dropout, bias, add_bias_kv, 
            add_zero_attn, kdim, vdim, batch_first
        )

    def forward(
        self, 
        query: torch.Tensor,
        key: torch.Tensor, 
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[torch.Tensor] = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
        sample_indices: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for Fast Nyström Attention.
        
        Args:
            query: Query tensor
            key: Key tensor (typically identical to query for self-attention)
            value: Value tensor (typically identical to query for self-attention)
            key_padding_mask: Mask for keys per batch to indicate padding
            need_weights: Whether to return attention weights
            attn_mask: Mask to prevent attention to certain positions
            average_attn_weights: Whether to average attention weights over heads
            is_causal: Whether to apply causal mask
            sample_indices: Indices of landmark points for Nyström approximation
            
        Returns:
            - Output tensor
            - Attention weights if need_weights is True, otherwise None
        """
        # Default to nn.MultiheadAttention if sample_indices is not provided
        if sample_indices is None:
            return super().forward(
                query, key, value, key_padding_mask, need_weights, attn_mask,
                average_attn_weights, is_causal
            )

        # Reshape to [B, N, D] if batch_first=False
        if not self.batch_first:
            query, key, value = query.transpose(0, 1), key.transpose(0, 1), value.transpose(0, 1)

        # Project query, key, value
        qkv = F.linear(query, self.in_proj_weight, self.in_proj_bias)
        query, key, value = rearrange(qkv, "b n (qkv h d) -> qkv b h n d", qkv=3, h=self.num_heads)
        
        x = fast_nystrom_attention(
            query,
            key,
            value,
            sample_indices,
            attn_mask=attn_mask,
            dropout_p=self.dropout,
            is_causal=is_causal,
        )
        x = rearrange(x, "b h n d -> b n (h d)")    

        # Project output
        x = F.linear(x, self.out_proj.weight, self.out_proj.bias)

        # Reshape back if batch_first=False
        if not self.batch_first:
            x = x.transpose(0, 1)
        
        return x, None