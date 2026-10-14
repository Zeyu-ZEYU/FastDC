"""PyTorch reference of the paged cache writes and of attention on the compressed paged cache.

These functions define the semantics that the Triton kernels in this package implement; tests compare
the kernels against them.

Shapes (T = tokens of the step, B = requests, H = query heads, Hkv = KV heads, g = H / Hkv):
    q             [T, H, Wq]   compressed queries; columns beyond a head's K width must be zero
    k_new, v_new  [T, Hkv, W]  values to write; only the first dims[h] columns of head h are stored
    pages         [num_blocks, page_elems]  one layer of the cache (any 2-byte dtype)
    slot_mapping  [T] int64    block * block_size + offset, or -1 to skip the token
    block_table   [B, max_blocks] int32
    query_start_loc [B + 1] int32 (cumulative query lengths), seq_lens [B] int32 (context + new tokens)
    out           [T, H, Wo]   attention output in compressed V coordinates; columns at and beyond a head's
                               V width are written as zero

Query token i of request b sits at position seq_lens[b] - q_len[b] + i and attends causally to cached
positions 0 .. its own position (its own K/V are written before attention).
"""

from __future__ import annotations

import torch

from fastdc.kernels.layout import LayerLayout, TileLayout


def write_tiles(pages: torch.Tensor, tiles: TileLayout, x: torch.Tensor, slot_mapping: torch.Tensor) -> None:
    keep = slot_mapping >= 0
    slots = slot_mapping[keep]
    x = x[keep]
    blocks, offs = slots // tiles.block_size, slots % tiles.block_size
    for h in range(tiles.num_heads):
        d = int(tiles.dims[h])
        base = blocks * tiles.page_elems + int(tiles.offsets[h]) + offs * d
        idx = base.unsqueeze(1) + torch.arange(d, device=pages.device)
        pages.view(-1)[idx.view(-1)] = x[:, h, :d].reshape(-1).to(pages.dtype)


def write_kv(pages: torch.Tensor, layout: LayerLayout, k_new: torch.Tensor, v_new: torch.Tensor,
             slot_mapping: torch.Tensor) -> None:
    write_tiles(pages, layout.k, k_new, slot_mapping)
    write_tiles(pages, layout.v, v_new, slot_mapping)


def gather_tiles(pages: torch.Tensor, tiles: TileLayout, blocks: torch.Tensor, length: int, width: int) -> torch.Tensor:
    """Tokens 0..length-1 of one request -> [Hkv, length, width] (zero-padded beyond each head's width)."""
    pos = torch.arange(length, device=pages.device)
    phys = blocks[pos // tiles.block_size].long()
    offs = pos % tiles.block_size
    out = torch.zeros(tiles.num_heads, length, width, dtype=pages.dtype, device=pages.device)
    for h in range(tiles.num_heads):
        d = int(tiles.dims[h])
        base = phys * tiles.page_elems + int(tiles.offsets[h]) + offs * d
        idx = base.unsqueeze(1) + torch.arange(d, device=pages.device)
        out[h, :, :d] = pages.view(-1)[idx.view(-1)].view(length, d)
    return out


def paged_attention(q: torch.Tensor, k_pages: torch.Tensor, k_tiles: TileLayout, v_pages: torch.Tensor,
                    v_tiles: TileLayout, block_table: torch.Tensor, query_start_loc: torch.Tensor,
                    seq_lens: torch.Tensor, scale: float, out: torch.Tensor,
                    k_block_table: torch.Tensor | None = None) -> torch.Tensor:
    """Causal attention of every request's new tokens on the paged compressed cache (see module docstring).

    K pages are addressed through `k_block_table` when given (e.g. a temporary buffer of decompressed keys),
    otherwise through `block_table` like the V pages.
    """
    k_table = block_table if k_block_table is None else k_block_table
    T, H, _ = q.shape
    hkv = k_tiles.num_heads
    g = H // hkv
    wk, wv = k_tiles.max_dim, v_tiles.max_dim
    out.zero_()
    qsl = query_start_loc.tolist()
    lens = seq_lens.tolist()
    for b in range(len(lens)):
        q0, q1, L = qsl[b], qsl[b + 1], lens[b]
        n = q1 - q0
        if n == 0:
            continue
        k = gather_tiles(k_pages, k_tiles, k_table[b], L, wk).float()  # [Hkv, L, wk]
        v = gather_tiles(v_pages, v_tiles, block_table[b], L, wv).float()
        qb = q[q0:q1, :, :wk].float().transpose(0, 1).reshape(hkv, g, n, wk)
        scores = torch.einsum("hgnw,hlw->hgnl", qb, k) * scale
        qpos = torch.arange(L - n, L, device=q.device)
        mask = torch.arange(L, device=q.device).unsqueeze(0) > qpos.unsqueeze(1)  # [n, L]
        scores = scores.masked_fill(mask, float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        res = torch.einsum("hgnl,hlw->hgnw", probs, v).reshape(H, n, wv).transpose(0, 1)  # [n, H, wv]
        out[q0:q1, :, :wv] = res.to(out.dtype)
    vdims = v_tiles.dims.long().repeat_interleave(g)  # per query head
    cols = torch.arange(out.shape[-1], device=out.device)
    out.masked_fill_(cols.view(1, 1, -1) >= vdims.view(1, -1, 1), 0)
    return out
