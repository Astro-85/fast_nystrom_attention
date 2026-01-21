"""Utility helpers for normalizing generation latency measurements.

These helpers estimate how the cost of decoding grows with the context length
and provide lightweight normalization for per-token throughput comparisons.
"""
from __future__ import annotations

from typing import Optional

_MIN_CONTEXT_LENGTH = 1
_EPS = 1e-6


def average_context_length(num_starting_tokens: int, num_generated_tokens: int) -> Optional[float]:
    """Estimate the average context length seen during autoregressive decoding.

    Each newly generated token attends to the starting context plus all prior
    generations. The average context length across *n* generated tokens is thus
    ``start + (n - 1) / 2``.
    """

    if num_generated_tokens <= 0:
        return None

    start = max(num_starting_tokens, _MIN_CONTEXT_LENGTH)
    avg_increment = max(num_generated_tokens - 1, 0) / 2.0
    return float(start + avg_increment)


def context_normalization_factor(
    num_starting_tokens: int, num_generated_tokens: int
) -> Optional[float]:
    """Return the multiplicative slowdown caused by context growth.

    A factor of 1.0 means no adjustment. Returns ``None`` when no new tokens
    were generated. Values > 1 indicate that later tokens were decoded under
    longer contexts.
    """

    avg_context = average_context_length(num_starting_tokens, num_generated_tokens)
    if avg_context is None:
        return None

    base_context = max(num_starting_tokens, _MIN_CONTEXT_LENGTH)
    factor = avg_context / base_context
    return max(factor, _EPS)


def normalize_latency(
    latency_s: Optional[float], num_starting_tokens: int, num_generated_tokens: int
) -> Optional[float]:
    """Scale latency to approximate a fixed-context decode workload."""

    if latency_s is None:
        return None

    factor = context_normalization_factor(num_starting_tokens, num_generated_tokens)
    if factor is None or factor <= _EPS:
        return None
    return latency_s / factor
