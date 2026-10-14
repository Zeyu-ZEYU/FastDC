"""Triton scatter of new K and V rows into the paged compressed cache (see fastdc.kernels.layout)."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from fastdc.kernels._common import device_tables, tile_info
from fastdc.kernels.layout import LayerLayout

BLOCK_T = 16  # tokens per program


@triton.jit
def _write_tile(pages_ptr, x_ptr, slots, tok, tok_ok, head, dims_ptr, offs_ptr, page_elems, stride_xt, stride_xh,
                BS: tl.constexpr, BLOCK_D: tl.constexpr):
    width = tl.load(dims_ptr + head)
    base = tl.load(offs_ptr + head)
    d = tl.arange(0, BLOCK_D)
    mask = tok_ok[:, None] & (d[None, :] < width)
    x = tl.load(x_ptr + tok.to(tl.int64)[:, None] * stride_xt + head * stride_xh + d[None, :], mask=mask)
    dst = (slots // BS) * page_elems + base + (slots % BS) * width
    tl.store(pages_ptr + dst[:, None] + d[None, :], x.to(pages_ptr.dtype.element_ty), mask=mask)


@triton.jit(do_not_specialize=["num_tokens", "stride_kt", "stride_vt"])
def _write_kv_kernel(pages_ptr, k_ptr, v_ptr, slot_ptr, k_dims, k_offs, v_dims, v_offs, k_page_elems, v_page_elems,
                     num_tokens, stride_kt, stride_kh, stride_vt, stride_vh,
                     K_BS: tl.constexpr, V_BS: tl.constexpr, BLOCK_T: tl.constexpr, BLOCK_DK: tl.constexpr,
                     BLOCK_DV: tl.constexpr):
    tok = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    head = tl.program_id(1)
    in_range = tok < num_tokens
    slots = tl.load(slot_ptr + tok, mask=in_range, other=-1)
    tok_ok = in_range & (slots >= 0)
    _write_tile(pages_ptr, k_ptr, slots, tok, tok_ok, head, k_dims, k_offs, k_page_elems, stride_kt, stride_kh,
                K_BS, BLOCK_DK)
    _write_tile(pages_ptr, v_ptr, slots, tok, tok_ok, head, v_dims, v_offs, v_page_elems, stride_vt, stride_vh,
                V_BS, BLOCK_DV)


def write_kv(pages: torch.Tensor, layout: LayerLayout, k_new: torch.Tensor, v_new: torch.Tensor,
             slot_mapping: torch.Tensor) -> None:
    """Store k_new / v_new [T, Hkv, W] at slot_mapping [T] (block * block_size + offset; -1 skips a token).

    Only the first dims[h] columns of head h are stored, at the head's place in its page.
    """
    k_info, v_info = tile_info(layout.k), tile_info(layout.v)
    num_tokens = slot_mapping.numel()
    if k_new.dim() != 3 or v_new.dim() != 3:
        raise ValueError("k_new and v_new must be [num_tokens, num_kv_heads, width]")
    if k_new.shape[0] < num_tokens or v_new.shape[0] < num_tokens:
        raise ValueError("k_new and v_new need a row for every entry of slot_mapping")
    if k_new.shape[1] != k_info.num_heads or v_new.shape[1] != v_info.num_heads:
        raise ValueError("k_new and v_new must have one entry per KV head of the layout")
    if k_new.shape[2] < k_info.max_dim or v_new.shape[2] < v_info.max_dim:
        raise ValueError("k_new and v_new must be at least as wide as the widest head")
    if num_tokens == 0:
        return
    if k_new.stride(-1) != 1:
        k_new = k_new.contiguous()
    if v_new.stride(-1) != 1:
        v_new = v_new.contiguous()
    slot_mapping = slot_mapping.to(torch.int64)
    device = pages.device
    k_dims, k_offs = device_tables(layout.k, device)
    v_dims, v_offs = device_tables(layout.v, device)
    grid = (triton.cdiv(num_tokens, BLOCK_T), k_info.num_heads)
    _write_kv_kernel[grid](pages, k_new, v_new, slot_mapping, k_dims, k_offs, v_dims, v_offs,
                           k_info.page_elems, v_info.page_elems, num_tokens,
                           k_new.stride(0), k_new.stride(1), v_new.stride(0), v_new.stride(1),
                           K_BS=k_info.block_size, V_BS=v_info.block_size, BLOCK_T=BLOCK_T,
                           BLOCK_DK=k_info.padded, BLOCK_DV=v_info.padded, num_warps=4)
