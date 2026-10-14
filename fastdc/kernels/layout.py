"""Paged layout of the compressed KV cache of one layer.

A page (vLLM block) holds `block_size` tokens of every KV head of the layer. Inside a page, each KV head
occupies two contiguous row-major tiles at its own widths: K [block_size, k_dims[h]] followed by
V [block_size, v_dims[h]]. `offsets` locate the tiles (in elements of the cache dtype). The cache stores
the unpadded widths; kernels pad to the tensor-core tile size internally.

The element at (block b, token t, head h, dimension d) of K lives at
    b * page_elems + k_offsets[h] + t * k_dims[h] + d
and the same holds for V with v_offsets / v_dims.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

PAGE_ALIGN = 8  # elements; keeps every page 16-byte aligned for 2-byte dtypes


@dataclass
class TileLayout:
    """Placement of one tensor kind (K or V) of every KV head inside a page."""

    dims: torch.Tensor  # [Hkv] int32
    offsets: torch.Tensor  # [Hkv] int64
    page_elems: int
    block_size: int

    @property
    def num_heads(self) -> int:
        return self.dims.numel()

    @property
    def max_dim(self) -> int:
        return int(self.dims.max())

    def to(self, device) -> "TileLayout":
        return TileLayout(self.dims.to(device), self.offsets.to(device), self.page_elems, self.block_size)


@dataclass
class LayerLayout:
    k: TileLayout
    v: TileLayout

    @property
    def page_elems(self) -> int:
        return self.k.page_elems

    @property
    def block_size(self) -> int:
        return self.k.block_size

    def to(self, device) -> "LayerLayout":
        return LayerLayout(self.k.to(device), self.v.to(device))


def layer_layout(k_dims, v_dims, block_size: int, device=None) -> LayerLayout:
    """Interleaved per-head tiles: K_0 V_0 K_1 V_1 ... (see the module docstring)."""
    k_dims = [int(x) for x in k_dims]
    v_dims = [int(x) for x in v_dims]
    k_off, v_off, cursor = [], [], 0
    for kd, vd in zip(k_dims, v_dims):
        k_off.append(cursor)
        cursor += block_size * kd
        v_off.append(cursor)
        cursor += block_size * vd
    page = (cursor + PAGE_ALIGN - 1) // PAGE_ALIGN * PAGE_ALIGN
    mk = lambda xs, dt: torch.tensor(xs, dtype=dt, device=device)
    return LayerLayout(TileLayout(mk(k_dims, torch.int32), mk(k_off, torch.int64), page, block_size),
                       TileLayout(mk(v_dims, torch.int32), mk(v_off, torch.int64), page, block_size))


def full_layer_layout(num_kv_heads: int, head_dim: int, block_size: int, device=None) -> LayerLayout:
    return layer_layout([head_dim] * num_kv_heads, [head_dim] * num_kv_heads, block_size, device)
