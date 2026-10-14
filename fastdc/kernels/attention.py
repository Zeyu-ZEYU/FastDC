"""Triton paged attention on the compressed KV cache (decode, chunked prefill and mixed batches).

Work is split into tasks (see fastdc.kernels.schedule). A task is (request, KV head, first token, end token)
and covers every query head of the KV head's GQA group for those tokens: its g * (end - first) rows are the
(token, query head) pairs in token-major order, so the group shares one pass over its K and V tiles. Rows
are processed in tiles; tasks with at most SMALL_M rows (single-token requests) use SMALL_M-row tiles and
larger tasks use LARGE_M-row tiles. Every row tile streams the keys visible to its last row in blocks of
key positions with an fp32 online softmax; only key blocks that cross the causal diagonal are masked.

Heads have their own widths. Tiles are as wide as the layer's widest head rounded up to a power of two,
and the head dimension is processed in chunks (LARGE_CHUNK or SMALL_CHUNK columns): loads are masked at
the head's width, so only the stored columns are read, and the matrix products of chunks at or beyond a
head's K (or V) width are skipped, so the tensor-core work of a task follows the head's widths rounded up
to the chunk. Output columns at and beyond a head's V width are written as zero, and so are the rows of
tokens that belong to no request.

A launch either runs one program per task, or one persistent program per worker that walks its own task
list.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import triton
import triton.language as tl

from fastdc.kernels._common import device_tables, tile_info
from fastdc.kernels.layout import LayerLayout, TileLayout

if TYPE_CHECKING:
    from fastdc.kernels.schedule import Schedule

SMALL_M, SMALL_N, SMALL_CHUNK = 16, 64, 32  # rows, key positions and head-dimension chunk of small tiles
LARGE_N, LARGE_CHUNK = 32, 16  # key positions and head-dimension chunk of large tiles
NUM_WARPS = 4
ZERO_CHUNK = tl.constexpr(32)  # columns per store when zeroing output rows


def large_tile_rows(group_size: int) -> int:
    """Rows per tile of multi-token tasks: 128, or 64 without GQA."""
    return 64 if group_size == 1 else 128


@triton.jit
def _kv_block(acc, m_i, l_i, q_chunks, q_pos, n0, kv_end,
              k_pages, v_pages, k_table, v_table, k_page_elems, v_page_elems, k_off, v_off, kd, vd, scale_log2,
              BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, K_BS: tl.constexpr, V_BS: tl.constexpr,
              CK: tl.constexpr, NCK: tl.constexpr, CV: tl.constexpr, NCV: tl.constexpr,
              SAME_ADDR: tl.constexpr, CAUSAL: tl.constexpr):
    """One online-softmax step over key positions n0 .. n0 + BLOCK_N - 1."""
    offs_n = n0 + tl.arange(0, BLOCK_N)
    n_ok = offs_n < kv_end
    k_blk = tl.load(k_table + offs_n // K_BS, mask=n_ok, other=0).to(tl.int64)
    k_row = k_blk * k_page_elems + k_off + (offs_n % K_BS) * kd
    if SAME_ADDR:
        v_blk = k_blk
    else:
        v_blk = tl.load(v_table + offs_n // V_BS, mask=n_ok, other=0).to(tl.int64)
    v_row = v_blk * v_page_elems + v_off + (offs_n % V_BS) * vd

    # issue every load of the block before computing; chunks beyond a head's width are fully masked
    k_chunks = ()
    for c in tl.static_range(NCK):
        d = c * CK + tl.arange(0, CK)
        k_chunks = k_chunks + (tl.load(k_pages + k_row[None, :] + d[:, None],
                                       mask=(d[:, None] < kd) & n_ok[None, :], other=0.0),)
    v_chunks = ()
    for c in tl.static_range(NCV):
        d = c * CV + tl.arange(0, CV)
        v_chunks = v_chunks + (tl.load(v_pages + v_row[:, None] + d[None, :],
                                       mask=n_ok[:, None] & (d[None, :] < vd), other=0.0),)

    s = tl.zeros([BLOCK_M, BLOCK_N], tl.float32)
    for c in tl.static_range(NCK):
        if c * CK < kd:
            s = tl.dot(q_chunks[c], k_chunks[c], s)
    if CAUSAL:
        s = tl.where(offs_n[None, :] <= q_pos[:, None], s, float("-inf"))
    m_new = tl.maximum(m_i, tl.max(s, 1) * scale_log2)
    alpha = tl.exp2(m_i - m_new)
    p = tl.exp2(s * scale_log2 - m_new[:, None])
    l_i = l_i * alpha + tl.sum(p, 1)
    p = p.to(v_pages.dtype.element_ty)

    new_acc = ()
    for c in tl.static_range(NCV):
        a = acc[c]
        if c * CV < vd:
            a = tl.dot(p, v_chunks[c], a * alpha[:, None])
        new_acc = new_acc + (a,)
    return new_acc, m_new, l_i


@triton.jit
def _attend_rows(q_ptr, out_ptr, k_pages, v_pages, k_table, v_table, k_page_elems, v_page_elems,
                 k_off, v_off, kd, vd, head0, q_start, ctx, row_begin, row_end, scale_log2,
                 stride_qt, stride_qh, stride_ot, stride_oh, out_width,
                 G: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, K_BS: tl.constexpr,
                 V_BS: tl.constexpr, CK: tl.constexpr, NCK: tl.constexpr, CV: tl.constexpr, NCV: tl.constexpr,
                 SAME_ADDR: tl.constexpr):
    """Attention of task rows row_begin .. row_begin + BLOCK_M - 1 (clipped to row_end)."""
    rows = row_begin + tl.arange(0, BLOCK_M)
    row_ok = rows < row_end
    tok = rows // G
    head = head0 + rows % G
    q_pos = ctx + tok
    q_base = (q_start + tok).to(tl.int64) * stride_qt + head * stride_qh
    q_chunks = ()
    for c in tl.static_range(NCK):
        d = c * CK + tl.arange(0, CK)
        qc = tl.load(q_ptr + q_base[:, None] + d[None, :], mask=row_ok[:, None] & (d[None, :] < kd), other=0.0)
        q_chunks = q_chunks + (qc.to(k_pages.dtype.element_ty),)
    acc = ()
    for _ in tl.static_range(NCV):
        acc = acc + (tl.zeros([BLOCK_M, CV], tl.float32),)
    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)

    last_tok = (tl.minimum(row_begin + BLOCK_M, row_end) - 1) // G
    kv_end = ctx + last_tok + 1  # keys visible to the last row
    kv_open = (ctx + row_begin // G + 1) // BLOCK_N * BLOCK_N  # whole blocks visible to every row
    for n0 in range(0, kv_open, BLOCK_N):
        acc, m_i, l_i = _kv_block(acc, m_i, l_i, q_chunks, q_pos, n0, kv_end, k_pages, v_pages, k_table, v_table,
                                  k_page_elems, v_page_elems, k_off, v_off, kd, vd, scale_log2,
                                  BLOCK_M, BLOCK_N, K_BS, V_BS, CK, NCK, CV, NCV, SAME_ADDR, False)
    for n0 in range(kv_open, kv_end, BLOCK_N):
        acc, m_i, l_i = _kv_block(acc, m_i, l_i, q_chunks, q_pos, n0, kv_end, k_pages, v_pages, k_table, v_table,
                                  k_page_elems, v_page_elems, k_off, v_off, kd, vd, scale_log2,
                                  BLOCK_M, BLOCK_N, K_BS, V_BS, CK, NCK, CV, NCV, SAME_ADDR, True)

    inv_l = 1.0 / tl.where(l_i > 0, l_i, 1.0)
    o_base = (q_start + tok).to(tl.int64) * stride_ot + head * stride_oh
    for c in tl.static_range(NCV):
        d = c * CV + tl.arange(0, CV)
        tl.store(out_ptr + o_base[:, None] + d[None, :], (acc[c] * inv_l[:, None]).to(out_ptr.dtype.element_ty),
                 mask=row_ok[:, None] & (d[None, :] < out_width))
    zeros = tl.zeros([BLOCK_M, CV], out_ptr.dtype.element_ty)
    for c0 in range(NCV * CV, out_width, CV):
        d = c0 + tl.arange(0, CV)
        tl.store(out_ptr + o_base[:, None] + d[None, :], zeros, mask=row_ok[:, None] & (d[None, :] < out_width))


@triton.jit
def _zero_rows(out_ptr, first, last, step, num_heads, out_width, stride_ot, stride_oh, CV: tl.constexpr):
    """Write zeros to tokens first, first + step, ... < last (all heads and columns)."""
    zeros = tl.zeros([CV], out_ptr.dtype.element_ty)
    for t in range(first, last, step):
        for h in range(num_heads):
            base = t.to(tl.int64) * stride_ot + h * stride_oh
            for c0 in range(0, out_width, CV):
                d = c0 + tl.arange(0, CV)
                tl.store(out_ptr + base + d, zeros, mask=d < out_width)


@triton.jit(do_not_specialize=["task_base", "bound_base", "num_tokens", "num_reqs", "stride_qt", "stride_ot",
                              "stride_bt", "stride_kbt"])
def _paged_attention_kernel(
        q_ptr, out_ptr, k_pages, v_pages, block_table, k_block_table, k_dims, k_offs, v_dims, v_offs,
        qsl_ptr, seq_lens_ptr, tasks_ptr, bounds_ptr, task_base, bound_base, scale_log2,
        num_tokens, num_reqs, num_heads, out_width, stride_qt, stride_qh, stride_ot, stride_oh,
        stride_bt, stride_kbt, k_page_elems, v_page_elems,
        G: tl.constexpr, K_BS: tl.constexpr, V_BS: tl.constexpr,
        S_CK: tl.constexpr, S_NCK: tl.constexpr, S_CV: tl.constexpr, S_NCV: tl.constexpr,
        L_CK: tl.constexpr, L_NCK: tl.constexpr, L_CV: tl.constexpr, L_NCV: tl.constexpr,
        SMALL_M: tl.constexpr, SMALL_N: tl.constexpr, LARGE_M: tl.constexpr, LARGE_N: tl.constexpr,
        HAS_SMALL: tl.constexpr, HAS_LARGE: tl.constexpr, SAME_ADDR: tl.constexpr, PERSISTENT: tl.constexpr):
    pid = tl.program_id(0)
    tasks_ptr += task_base
    bounds_ptr += bound_base
    if PERSISTENT:
        first = tl.load(bounds_ptr + pid)
        last = tl.load(bounds_ptr + pid + 1)
    else:
        first = pid
        last = tl.minimum(pid + 1, tl.load(bounds_ptr + 1))
    for t in range(first, last):
        task = tasks_ptr + t.to(tl.int64) * 4
        req = tl.load(task)
        kvh = tl.load(task + 1)
        r_begin = tl.load(task + 2) * G
        r_end = tl.load(task + 3) * G
        q_start = tl.load(qsl_ptr + req)
        q_len = tl.load(qsl_ptr + req + 1) - q_start
        ctx = tl.load(seq_lens_ptr + req) - q_len
        kd = tl.load(k_dims + kvh)
        vd = tl.load(v_dims + kvh)
        k_off = tl.load(k_offs + kvh)
        v_off = tl.load(v_offs + kvh)
        v_table = block_table + req.to(tl.int64) * stride_bt
        k_table = k_block_table + req.to(tl.int64) * stride_kbt
        small = r_end - r_begin <= SMALL_M
        if HAS_SMALL and small:
            _attend_rows(q_ptr, out_ptr, k_pages, v_pages, k_table, v_table, k_page_elems, v_page_elems,
                         k_off, v_off, kd, vd, kvh * G, q_start, ctx, r_begin, r_end, scale_log2,
                         stride_qt, stride_qh, stride_ot, stride_oh, out_width,
                         G, SMALL_M, SMALL_N, K_BS, V_BS, S_CK, S_NCK, S_CV, S_NCV, SAME_ADDR)
        if HAS_LARGE and not small:
            for row_begin in range(r_begin, r_end, LARGE_M):
                _attend_rows(q_ptr, out_ptr, k_pages, v_pages, k_table, v_table, k_page_elems, v_page_elems,
                             k_off, v_off, kd, vd, kvh * G, q_start, ctx, row_begin, r_end, scale_log2,
                             stride_qt, stride_qh, stride_ot, stride_oh, out_width,
                             G, LARGE_M, LARGE_N, K_BS, V_BS, L_CK, L_NCK, L_CV, L_NCV, SAME_ADDR)
    # tokens that belong to no request are written as zeros
    step = tl.num_programs(0)
    _zero_rows(out_ptr, pid, tl.load(qsl_ptr), step, num_heads, out_width, stride_ot, stride_oh, ZERO_CHUNK)
    _zero_rows(out_ptr, tl.load(qsl_ptr + num_reqs) + pid, num_tokens, step, num_heads, out_width,
               stride_ot, stride_oh, ZERO_CHUNK)


def paged_attention(q: torch.Tensor, k_pages: torch.Tensor, k_tiles: TileLayout, v_pages: torch.Tensor,
                    v_tiles: TileLayout, block_table: torch.Tensor, query_start_loc: torch.Tensor,
                    seq_lens: torch.Tensor, scale: float, out: torch.Tensor,
                    k_block_table: torch.Tensor | None = None, schedule: Schedule | None = None,
                    balance: bool = True) -> torch.Tensor:
    """Causal attention of every request's new tokens on the paged compressed cache; returns `out`.

    Semantics follow fastdc.kernels.reference.paged_attention: q [T, H, Wq], pages [num_blocks, page_elems]
    of a 2-byte dtype, int32 query_start_loc [B + 1] and seq_lens [B], out [T, H, Wo]. `schedule` is this
    layer's entry of build_schedules for the batch (it must describe the same K and V widths); without
    one, a schedule is built here with `balance`, which reads batch sizes from the device.
    """
    from fastdc.kernels.schedule import build_schedules

    if q.dim() != 3 or out.dim() != 3 or out.shape[:2] != q.shape[:2]:
        raise ValueError("q and out must be [num_tokens, num_heads, width] with matching first two dimensions")
    if out.stride(-1) != 1:
        raise ValueError("out must be contiguous in its last dimension")
    if q.stride(-1) != 1:
        q = q.contiguous()
    k_info, v_info = tile_info(k_tiles), tile_info(v_tiles)
    num_tokens, num_heads, _ = q.shape
    hkv = k_info.num_heads
    if v_info.num_heads != hkv or num_heads % hkv != 0:
        raise ValueError("the number of query heads must be a multiple of the number of KV heads")
    if q.shape[-1] < k_info.max_dim or out.shape[-1] < v_info.max_dim:
        raise ValueError("q and out must be at least as wide as the widest K and V head")
    if k_pages.element_size() != 2 or v_pages.element_size() != 2:
        raise ValueError("cache pages must use a 2-byte dtype")
    if query_start_loc.dtype != torch.int32 or seq_lens.dtype != torch.int32:
        raise ValueError("query_start_loc and seq_lens must be int32")
    group = num_heads // hkv
    if schedule is None:
        schedule = build_schedules([LayerLayout(k_tiles, v_tiles)], query_start_loc, seq_lens, num_heads,
                                   balance=balance)[0]
    if schedule.group_size != group:
        raise ValueError("the schedule was built for a different number of query heads")

    device = q.device
    k_dims, k_offs = device_tables(k_tiles, device)
    v_dims, v_offs = device_tables(v_tiles, device)
    block_dk, block_dv = k_info.padded, v_info.padded
    s_ck, s_cv = min(SMALL_CHUNK, block_dk), min(SMALL_CHUNK, block_dv)
    l_ck, l_cv = min(LARGE_CHUNK, block_dk), min(LARGE_CHUNK, block_dv)
    k_table = block_table if k_block_table is None else k_block_table
    same_addr = k_block_table is None and k_info.block_size == v_info.block_size
    _paged_attention_kernel[(schedule.num_programs,)](
        q, out, k_pages, v_pages, block_table, k_table, k_dims, k_offs, v_dims, v_offs,
        query_start_loc, seq_lens, schedule.tasks, schedule.bounds, schedule.layer * schedule.tasks.stride(0),
        schedule.layer * schedule.bounds.stride(0), float(scale) * math.log2(math.e),
        num_tokens, query_start_loc.numel() - 1, num_heads, out.shape[-1], q.stride(0), q.stride(1),
        out.stride(0), out.stride(1), block_table.stride(0), k_table.stride(0), k_info.page_elems, v_info.page_elems,
        G=group, K_BS=k_info.block_size, V_BS=v_info.block_size,
        S_CK=s_ck, S_NCK=block_dk // s_ck, S_CV=s_cv, S_NCV=block_dv // s_cv,
        L_CK=l_ck, L_NCK=block_dk // l_ck, L_CV=l_cv, L_NCV=block_dv // l_cv,
        SMALL_M=SMALL_M, SMALL_N=SMALL_N, LARGE_M=large_tile_rows(group), LARGE_N=LARGE_N,
        HAS_SMALL=schedule.has_small, HAS_LARGE=schedule.has_large, SAME_ADDR=same_addr,
        PERSISTENT=schedule.persistent, num_warps=NUM_WARPS, num_stages=1)
    return out
