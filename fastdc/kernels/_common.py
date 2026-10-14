"""Host-side helpers shared by the kernels: cached per-layout metadata and device properties.

Kernels need a few host integers about a layout (widths, padded tile widths) and device copies of its
per-head tables. Reading them from a GPU tensor synchronizes the device, so they are computed once per
layout tensor and cached. The cache holds a reference to the tensor it describes, so a cached entry can
only be returned for that same tensor object.
"""

from __future__ import annotations

import functools
from collections import OrderedDict
from dataclasses import dataclass

import torch
import triton

from fastdc.kernels.layout import TileLayout

MIN_TILE = 16  # smallest edge of a tensor-core tile
_CACHE_SIZE = 512


def padded_width(width: int) -> int:
    """Smallest power of two that is >= width and >= MIN_TILE."""
    return max(MIN_TILE, triton.next_power_of_2(int(width)))


@dataclass(frozen=True)
class TileInfo:
    """Host copy of the per-head widths of one TileLayout."""

    dims: tuple[int, ...]
    block_size: int
    page_elems: int

    @property
    def num_heads(self) -> int:
        return len(self.dims)

    @property
    def max_dim(self) -> int:
        return max(self.dims)

    @property
    def padded(self) -> int:
        return padded_width(self.max_dim)


class _IdentityCache:
    """LRU cache keyed by tensor identity; an entry is valid only for the exact tensor object it was built for."""

    def __init__(self, size: int = _CACHE_SIZE):
        self._entries: OrderedDict[tuple, tuple[tuple, object]] = OrderedDict()
        self._size = size

    def get(self, tensors: tuple, extra: tuple = ()):
        key = tuple(id(t) for t in tensors) + extra
        hit = self._entries.get(key)
        if hit is not None and all(a is b for a, b in zip(hit[0], tensors)):
            self._entries.move_to_end(key)
            return hit[1]
        return None

    def put(self, tensors: tuple, value, extra: tuple = ()):
        key = tuple(id(t) for t in tensors) + extra
        self._entries[key] = (tensors, value)
        if len(self._entries) > self._size:
            self._entries.popitem(last=False)
        return value


_tile_infos = _IdentityCache()
_device_tables = _IdentityCache()


def tile_info(tiles: TileLayout) -> TileInfo:
    """Host widths of `tiles` (synchronizes once per dims tensor when it lives on the GPU)."""
    info = _tile_infos.get((tiles.dims,), (tiles.block_size, tiles.page_elems))
    if info is None:
        dims = tuple(int(x) for x in tiles.dims.tolist())
        if not dims or min(dims) < 1:
            raise ValueError("every head needs a width >= 1")
        info = _tile_infos.put((tiles.dims,), TileInfo(dims, int(tiles.block_size), int(tiles.page_elems)),
                               (tiles.block_size, tiles.page_elems))
    return info


def device_tables(tiles: TileLayout, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """(dims int32, offsets int64) of `tiles` on `device`, contiguous."""
    extra = (str(device),)
    tables = _device_tables.get((tiles.dims, tiles.offsets), extra)
    if tables is None:
        dims = tiles.dims.to(device=device, dtype=torch.int32).contiguous()
        offsets = tiles.offsets.to(device=device, dtype=torch.int64).contiguous()
        tables = _device_tables.put((tiles.dims, tiles.offsets), (dims, offsets), extra)
    return tables


@functools.lru_cache(maxsize=None)
def _sm_count(index: int) -> int:
    return torch.cuda.get_device_properties(index).multi_processor_count


def num_sms(device: torch.device) -> int:
    """Number of streaming multiprocessors of a CUDA device."""
    device = torch.device(device)
    index = device.index if device.index is not None else torch.cuda.current_device()
    return _sm_count(index)
