"""This script implements token pruning for Llava, allowing for dynamic reduction of
the latent token space passed to the multi-modal projector. """
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn

def normalize_pruning_config(config: Dict) -> Dict:
    confg = dict(config) if config is not None else {}
    confg.setdefault("patch_dim", 256)
    confg.setdefault("embedding_dim", 512)
    confg.setdefault("pruning_num_latents", 32)
    confg.setdefault("pruning_num_layers", 1)
    confg.setdefault("pruning_num_heads", 8)
    confg.setdefault("pruning_mlp_ratio", 4.0)
    confg.setdefault("pruning_dropout", 0.0)
    confg.setdefault("codebook_size", 1024)
    confg.setdefault("commitment_loss_weight", 0.25)
    return confg



class LlavaNextCompressor(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.config = normalize_pruning_config(config)
        self.D = self.config["embedding_dim"]
        self.K = self.config["pruning_num_latents"]
        self.num_layers = self.config["pruning_num_layers"]
        self.num_heads = self.config["pruning_num_heads"]
        self.mlp_ratio = self.config["pruning_mlp_ratio"]
        self.dropout = self.config["pruning_dropout"]
        self.codebook_size = self.config["codebook_size"]

        if self.D % self.num_heads != 0:
            raise ValueError(f"Embedding dimension {self.D} must be divisible by number of heads {self.num_heads}")

        self.latent_tokens = nn.Parameter(torch.randn(1, self.K, self.D), requires_grad=True)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=self.D,
                nhead=self.num_heads,
                dim_feedforward=int(self.D * self.mlp_ratio),
                dropout=self.dropout,
                activation="gelu",
                batch_first=True
            )
            for _ in range(self.num_layers)
        ])

        self.norm = nn.LayerNorm(self.D)
        self.codebook = nn.Embedding(self.codebook_size, self.D)



    def forward(self, x):
        B, N, D = x.shape
        if D != self.D:
            raise ValueError(f"Input embedding dimension {D} does not match compressor config {self.D}")

        latents = self.latent_tokens.expand(B, -1, -1)
        tokens = torch.cat([x, latents], dim=1)
        for block in self.blocks:
            tokens = block(tokens)
            
        tokens = self.norm(tokens)
        latents = tokens[:, N:, :]
        flattened = latents.reshape(-1, self.D)
        distances = torch.cdist(flattened, self.codebook.weight)
        indices = torch.argmin(distances, dim=1)
        quantized = self.codebook(indices).view(B, self.K, self.D)
        quantized_st = latents + (quantized - latents).detach()

        beta = self.config["commitment_loss_weight"]

        codebook_loss = ((quantized - latents.detach())**2).mean()
        commitment_loss = ((latents - quantized.detach())**2).mean()
        vq_loss = codebook_loss + beta * commitment_loss


        return quantized_st, vq_loss

        



        
        

    