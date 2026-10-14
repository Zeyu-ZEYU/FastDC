"""Offline stage, step 3: compression rates from singular values.

The matrix energy removal rate r is the share of the squared singular values that a head discards.
Given r, a head keeps the smallest number of leading dimensions whose discarded tail carries at most
r of the energy (Eq. (1) of the paper); its compression rate is p = 1 - kept / head_dim. One shared r
gives every head and layer its own rate through its own spectrum.
"""

from __future__ import annotations

import torch


def kept_dims(sigma: torch.Tensor, r: float, minimum: int = 1) -> torch.Tensor:
    """sigma [..., D] (descending) -> number of kept dimensions [...] for energy removal rate r."""
    energy = sigma.double() ** 2
    total = energy.sum(-1, keepdim=True)
    # tail[k] = energy of dimensions k..D-1
    tail = energy.flip(-1).cumsum(-1).flip(-1)
    kept = (tail > r * total * (1 + 1e-12)).sum(-1)
    return kept.clamp(min=minimum, max=sigma.shape[-1]).long()


def uniform_dims(shape_prefix: tuple[int, ...], head_dim: int, p: float, minimum: int = 1) -> torch.Tensor:
    """The same number of kept dimensions for every head: round((1 - p) * D)."""
    kept = max(minimum, min(head_dim, round((1.0 - p) * head_dim)))
    return torch.full(shape_prefix, kept, dtype=torch.long)


def removed_energy(sigma: torch.Tensor, kept: torch.Tensor) -> torch.Tensor:
    """Share of the energy in the discarded dimensions, per head."""
    energy = sigma.double() ** 2
    idx = torch.arange(sigma.shape[-1])
    mask = idx >= kept.unsqueeze(-1)
    return (energy * mask).sum(-1) / energy.sum(-1)


def kv_compression_rate(k_dims: torch.Tensor, v_dims: torch.Tensor, head_dim: int) -> float:
    """Overall KV compression rate: 1 - stored KV widths / uncompressed KV widths."""
    return 1.0 - float(k_dims.sum() + v_dims.sum()) / (2 * head_dim * k_dims.numel())
