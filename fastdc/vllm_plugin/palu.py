"""Keys of the palu method: pre-RoPE low-rank latents in the cache, reconstructed before attention.

Writing: the keys arrive after RoPE, so they are first rotated back (RoPE^-1) and then projected onto
the kept W_K directions B (D x w per KV head): k_lat = RoPE^-1(k) B.
Reading (the decompression step of this baseline): every cache block a step attends to is reconstructed
as K = RoPE(k_lat B^T) at the block's token positions into a full-width key buffer.

Both directions have a PyTorch implementation (`attention="reference"`) and a Triton kernel with the
same semantics; the kernels cover full-dimension rotary embeddings, the PyTorch code any rotary dimension.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from fastdc.kernels.layout import TileLayout
from fastdc.vllm_plugin.rope import Rotary


@triton.jit
def _compress_kernel(k_ptr, pos_ptr, cos_sin_ptr, b1_ptr, b2_ptr, out_ptr, num_tokens, max_pos, stride_kt, stride_kh,
                     stride_ot, stride_oh, HALF: tl.constexpr, W: tl.constexpr, NEOX: tl.constexpr,
                     BLOCK_T: tl.constexpr):
    """k_lat[t, h] = RoPE^-1(k[t, h]) B_h for BLOCK_T tokens of one KV head."""
    head = tl.program_id(1)
    tok = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    ok = tok < num_tokens
    i = tl.arange(0, HALF)
    d1 = i if NEOX else 2 * i
    d2 = i + HALF if NEOX else 2 * i + 1
    base = k_ptr + tok.to(tl.int64)[:, None] * stride_kt + head * stride_kh
    x1 = tl.load(base + d1[None, :], mask=ok[:, None], other=0.0).to(tl.float32)
    x2 = tl.load(base + d2[None, :], mask=ok[:, None], other=0.0).to(tl.float32)
    pos = tl.minimum(tl.maximum(tl.load(pos_ptr + tok, mask=ok, other=0), 0), max_pos - 1)
    cs = cos_sin_ptr + pos.to(tl.int64)[:, None] * (2 * HALF) + i[None, :]
    cos = tl.load(cs).to(tl.float32)
    sin = tl.load(cs + HALF).to(tl.float32)
    norm = cos * cos + sin * sin
    p1 = (x1 * cos + x2 * sin) / norm
    p2 = (x2 * cos - x1 * sin) / norm
    w = tl.arange(0, W)
    rows = head * HALF * W + i[:, None] * W + w[None, :]
    lat = tl.dot(p1, tl.load(b1_ptr + rows)) + tl.dot(p2, tl.load(b2_ptr + rows))
    dst = out_ptr + tok.to(tl.int64)[:, None] * stride_ot + head * stride_oh + w[None, :]
    tl.store(dst, lat.to(out_ptr.dtype.element_ty), mask=ok[:, None])


@triton.jit
def _decompress_kernel(pages_ptr, page_elems, k_dims_ptr, k_offs_ptr, src_ptr, logical_ptr, cos_sin_ptr, max_pos,
                       b1_ptr, b2_ptr, out_ptr, HKV: tl.constexpr, BS: tl.constexpr, HALF: tl.constexpr,
                       W: tl.constexpr, NEOX: tl.constexpr):
    """Buffer block j, KV head h: K = RoPE(k_lat B_h^T) for the BS tokens of cache block src[j]."""
    j = tl.program_id(0)
    head = tl.program_id(1)
    width = tl.load(k_dims_ptr + head)
    offset = tl.load(k_offs_ptr + head)
    src = tl.load(src_ptr + j)
    t = tl.arange(0, BS)
    c = tl.arange(0, W)
    lat = tl.load(pages_ptr + src * page_elems + offset + t[:, None] * width + c[None, :], mask=c[None, :] < width,
                  other=0.0)
    i = tl.arange(0, HALF)
    rows = head * W * HALF + c[:, None] * HALF + i[None, :]
    x1 = tl.dot(lat, tl.load(b1_ptr + rows))
    x2 = tl.dot(lat, tl.load(b2_ptr + rows))
    pos = tl.minimum(tl.load(logical_ptr + j) * BS + t, max_pos - 1)
    cs = cos_sin_ptr + pos.to(tl.int64)[:, None] * (2 * HALF) + i[None, :]
    cos = tl.load(cs).to(tl.float32)
    sin = tl.load(cs + HALF).to(tl.float32)
    d1 = i if NEOX else 2 * i
    d2 = i + HALF if NEOX else 2 * i + 1
    dst = out_ptr + j.to(tl.int64) * (HKV * BS * 2 * HALF) + head * (BS * 2 * HALF) + t[:, None] * (2 * HALF)
    tl.store(dst + d1[None, :], (x1 * cos - x2 * sin).to(out_ptr.dtype.element_ty))
    tl.store(dst + d2[None, :], (x2 * cos + x1 * sin).to(out_ptr.dtype.element_ty))


class PaluKeys:
    """Key compression and reconstruction of one layer with the kept W_K basis (zero beyond a head's width)."""

    def __init__(self, basis: torch.Tensor, rotary: Rotary, dtype: torch.dtype, use_kernels: bool):
        self.rotary = rotary
        self.dtype = dtype
        num_kv, d, w = basis.shape
        self.basis = basis.float()  # [Hkv, D, W]
        self.basis_t = basis.transpose(1, 2).to(dtype).contiguous()  # [Hkv, W, D]
        self.use_kernels = use_kernels and rotary.rotary_dim == d and d >= 32 and d & (d - 1) == 0
        if self.use_kernels:
            half = d // 2
            idx = torch.arange(half, device=basis.device)
            first, second = (idx, idx + half) if rotary.neox else (2 * idx, 2 * idx + 1)
            self.width = max(16, triton.next_power_of_2(w))
            padded = torch.zeros(num_kv, d, self.width, device=basis.device)
            padded[..., :w] = self.basis
            # rows of the basis by rotation pair (compression) and columns by rotation pair (reconstruction)
            self.c1 = padded[:, first].contiguous()  # [Hkv, D/2, Wp] float32
            self.c2 = padded[:, second].contiguous()
            self.d1 = padded[:, first].transpose(1, 2).to(dtype).contiguous()  # [Hkv, Wp, D/2]
            self.d2 = padded[:, second].transpose(1, 2).to(dtype).contiguous()

    def compress(self, k: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Post-RoPE keys [T, Hkv, D] -> latent keys [T, Hkv, >= W]."""
        if not self.use_kernels:
            cos, sin = self.rotary.cos_sin(positions)
            k_pre = self.rotary.rotate(k.float(), cos.unsqueeze(1), sin.unsqueeze(1), inverse=True)
            return torch.bmm(k_pre.transpose(0, 1), self.basis).transpose(0, 1).to(self.dtype)
        num_tokens, num_kv, d = k.shape
        out = torch.empty(num_tokens, num_kv, self.width, dtype=self.dtype, device=k.device)
        if k.stride(-1) != 1:
            k = k.contiguous()
        table = self.rotary.module.cos_sin_cache
        block_t = 16
        _compress_kernel[(triton.cdiv(num_tokens, block_t), num_kv)](
            k, positions, table, self.c1, self.c2, out, num_tokens, table.shape[0], k.stride(0), k.stride(1),
            out.stride(0), out.stride(1), HALF=d // 2, W=self.width, NEOX=self.rotary.neox, BLOCK_T=block_t,
            num_warps=4)
        return out

    def decompress(self, pages: torch.Tensor, tiles: TileLayout, k_dims: tuple[int, ...], k_offsets: list[int],
                   src_blocks: torch.Tensor, logical_blocks: torch.Tensor, buffer: torch.Tensor) -> None:
        """Reconstruct the post-RoPE keys of cache blocks `src_blocks` into buffer pages [0, m).

        Buffer page j receives cache block src_blocks[j], which is block logical_blocks[j] of its request, as
        tiles [Hkv][block_size, D]; `tiles`, `k_dims` and `k_offsets` describe the latent keys in the cache.
        """
        m = src_blocks.numel()
        num_kv, w, d = self.basis_t.shape
        bs = tiles.block_size
        if m == 0:
            return
        if self.use_kernels and bs & (bs - 1) == 0:
            table = self.rotary.module.cos_sin_cache
            _decompress_kernel[(m, num_kv)](
                pages, tiles.page_elems, tiles.dims, tiles.offsets, src_blocks, logical_blocks, table, table.shape[0],
                self.d1, self.d2, buffer, HKV=num_kv, BS=bs, HALF=d // 2, W=self.width, NEOX=self.rotary.neox,
                num_warps=4)
            return
        keys = buffer[:m].view(m, num_kv, bs, d)
        positions = logical_blocks.unsqueeze(1) * bs + torch.arange(bs, device=pages.device)
        cos, sin = self.rotary.cos_sin(positions)  # [m, bs, rot / 2]
        for h, (width, offset) in enumerate(zip(k_dims, k_offsets)):
            latent = pages[:, offset : offset + bs * width].index_select(0, src_blocks).view(m, bs, width)
            full = torch.matmul(latent, self.basis_t[h, :width])
            keys[:, h] = self.rotary.rotate(full.float(), cos, sin).to(keys.dtype)
