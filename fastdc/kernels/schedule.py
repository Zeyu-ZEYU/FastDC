"""Load-balanced task schedules for the paged attention kernel (fastdc.kernels.attention).

Tasks. A task is (request, KV head, first token, end token) and covers the g query heads of the KV head's
group for those tokens, so the g heads share one pass over the K and V tiles. The paper's row block of
beta_r = 64 query rows per head therefore becomes a block of 64 tokens of all g heads (64 * g packed rows).

Workload. A task attends to s_kv keys (the keys visible to its last token) and costs
    rows_padded * s_kv * d_sum,    d_sum = d_qk + d_v + c,
following the per-task workload 2 * beta_r * s_kv * (d_qk + d_v + 1) of a FlashAttention-style kernel.
rows_padded is the task's g * tokens rows rounded up to the kernel's row tiles, d_qk and d_v are the head's
K and V widths rounded up to the kernel's 16-column chunks, and c (NON_MATMUL_COST) is the per-key work
outside the two matrix products (softmax and addressing) in units of one head-dimension column. Counting
only the softmax FLOPs gives c = 1; the default is set between the decode and prefill values that
benchmarks/kernels/attention_bench.py --calibrate measures for this kernel, and can be recalibrated for
another GPU. A single-token (decode) request gives one task per KV head whose cost is proportional to
s_kv * d_sum.

Balanced schedule (balance=True), per layer, over N workers (persistent programs):
  Prefill (requests with more than one new token):
    1. Row scaling: head h uses blocks of 64 * d_sum_max / d_sum_h tokens, rounded to whole row tiles of
       the kernel (LARGE_M = 128 rows, or 64 without GQA, i.e. LARGE_M / g tokens), so that blocks of
       different heads carry the same work.
    2. Tasks are sorted by cost (descending) and assigned greedily to the least-loaded worker.
    3. The last round, the final (number of tasks mod N) tasks, is re-split: with wl = (sum of their
       cost) / N, every such task is cut into blocks of about wl / (cost per row) rows (whole row tiles),
       and the blocks are assigned in order to the least-loaded worker.
  Decode (single-token requests): one task per (request, KV head), sorted by cost (descending) and
    assigned greedily to the least-loaded worker (LPT).
  Each worker runs its prefill tasks, then its decode tasks. Sorting runs with torch.sort and the greedy
  assignment in a Triton kernel with one program per layer, so the schedules of all layers are built on
  the GPU in a few launches without synchronizing with the host. Worker loads are compared after their
  lowest mantissa bits are replaced by the worker's tie-break rank (a relative resolution of 2^-16 for up
  to 128 workers). Ties go to the lowest worker index for prefill tasks and to the highest for decode
  tasks, so that the two phases fill the workers from opposite ends. A worker's tasks are stored
  contiguously (CSR offsets in `bounds`).

Unbalanced schedule (balance=False): fixed blocks of 64 tokens, no scaling and no redistribution; tasks stay
in their natural order (request, KV head, block) and the kernel runs one program per task.
"""

from __future__ import annotations

import math
import operator
from collections.abc import Sequence
from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from fastdc.kernels._common import num_sms, tile_info
from fastdc.kernels.attention import LARGE_CHUNK, SMALL_M, large_tile_rows
from fastdc.kernels.layout import LayerLayout

ROW_BLOCK = 64  # beta_r: tokens per prefill task before scaling
WORKERS_PER_SM = 2  # persistent programs per SM (the kernel keeps two programs resident per SM)
NON_MATMUL_COST = 128  # c in d_sum, in head-dimension columns
MAX_WORKERS = 4096  # the worker rank takes the low 12 bits of a load's key


@dataclass(slots=True)
class Schedule:
    """Task assignment of one layer's attention for one batch.

    `tasks[layer]` holds rows (request, kv_head, first_token, end_token) of int32. A persistent schedule
    gives worker w the tasks bounds[layer, w] .. bounds[layer, w + 1] - 1; otherwise program i runs task i
    when i < bounds[layer, 1].
    """

    tasks: torch.Tensor  # [num_layers, capacity, 4] int32
    bounds: torch.Tensor  # [num_layers, num_programs + 1] (persistent) or [1, 2] int32
    layer: int
    num_programs: int
    persistent: bool
    group_size: int  # query heads per KV head
    has_small: bool  # tasks of at most SMALL_M rows may occur
    has_large: bool  # tasks of more than SMALL_M rows may occur

    def _host_bounds(self) -> list[int]:
        return self.bounds[self.layer, :self.num_programs + 1 if self.persistent else 2].tolist()

    def layer_tasks(self) -> list[tuple[int, int, int, int]]:
        """Tasks of this layer in storage order (host copy; synchronizes)."""
        bounds = self._host_bounds()
        return [tuple(t) for t in self.tasks[self.layer, bounds[0]:bounds[-1]].tolist()]

    def worker_tasks(self) -> list[list[tuple[int, int, int, int]]]:
        """Task list of every worker of a persistent schedule (host copy; synchronizes)."""
        bounds = self._host_bounds()
        tasks = [tuple(t) for t in self.tasks[self.layer, :bounds[-1]].tolist()]
        return [tasks[bounds[w]:bounds[w + 1]] for w in range(len(bounds) - 1)]


class ScheduleWorkspace:
    """Output buffers reused by successive build_schedules calls.

    Schedules built with a workspace stay valid until the next call that uses the same workspace. The
    buffers only grow, so once a call has sized them for the largest batch, their addresses no longer
    change (as replaying a CUDA graph that contains paged_attention requires).
    """

    def __init__(self):
        self._buffers: dict[str, torch.Tensor] = {}

    def buffer(self, name: str, shape: tuple[int, ...], dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        buf = self._buffers.get(name)
        if (buf is None or buf.dtype != dtype or buf.device != torch.device(device) or buf.dim() != len(shape)
                or any(have < need for have, need in zip(buf.shape, shape))):
            if buf is not None and buf.dim() == len(shape):
                shape = tuple(max(have, need) for have, need in zip(buf.shape, shape))
            buf = torch.empty(shape, dtype=dtype, device=device)
            self._buffers[name] = buf
        return buf


def head_cost(k_dim: int, v_dim: int) -> int:
    """d_sum of one head: K and V widths rounded up to the kernel's chunks plus NON_MATMUL_COST."""
    return -(-k_dim // LARGE_CHUNK) * LARGE_CHUNK + -(-v_dim // LARGE_CHUNK) * LARGE_CHUNK + NON_MATMUL_COST


def tile_tokens(group_size: int) -> int:
    """Tokens per row tile of the kernel's multi-token path."""
    return max(1, large_tile_rows(group_size) // group_size)


def row_block_sizes(k_dims: Sequence[int], v_dims: Sequence[int], group_size: int) -> list[int]:
    """Tokens per prefill block of every head of a layer: 64 * d_sum_max / d_sum_h in whole row tiles."""
    d_sum = [head_cost(k, v) for k, v in zip(k_dims, v_dims)]
    top = max(d_sum)
    unit = tile_tokens(group_size)
    return [unit * max(1, math.floor(ROW_BLOCK * top / d / unit + 0.5)) for d in d_sum]


@dataclass
class _LayerStack:
    layouts: tuple  # the LayerLayout objects described
    num_kv_heads: int
    min_block: int  # smallest scaled block over all layers and heads
    row_blocks: torch.Tensor  # [L, Hkv] int32 scaled tokens per block
    fixed_blocks: torch.Tensor  # [1, Hkv] int32, ROW_BLOCK everywhere
    d_sum: torch.Tensor  # [L, Hkv] float32


_stacks: dict[tuple, _LayerStack] = {}
_MAX_STACKS = 64


def _layer_stack(layouts: Sequence[LayerLayout], group_size: int, device: torch.device) -> _LayerStack:
    """Per-layer block sizes and head costs on `device`, cached by the identity of the layout objects."""
    key = (tuple(map(id, layouts)), group_size, str(device))
    stack = _stacks.get(key)
    if stack is not None and len(stack.layouts) == len(layouts) and all(map(operator.is_, stack.layouts, layouts)):
        return stack
    hkv = tile_info(layouts[0].k).num_heads
    blocks, d_sum = [], []
    for layout in layouts:
        k_dims, v_dims = tile_info(layout.k).dims, tile_info(layout.v).dims
        if len(k_dims) != hkv or len(v_dims) != hkv:
            raise ValueError("all layers must have the same number of KV heads")
        blocks.append(row_block_sizes(k_dims, v_dims, group_size))
        d_sum.append([head_cost(k, v) for k, v in zip(k_dims, v_dims)])
    stack = _LayerStack(
        layouts=tuple(layouts),
        num_kv_heads=hkv,
        min_block=min(min(b) for b in blocks),
        row_blocks=torch.tensor(blocks, dtype=torch.int32, device=device),
        fixed_blocks=torch.full((1, hkv), ROW_BLOCK, dtype=torch.int32, device=device),
        d_sum=torch.tensor(d_sum, dtype=torch.float32, device=device),
    )
    if len(_stacks) >= _MAX_STACKS:
        _stacks.pop(next(iter(_stacks)))
    _stacks[key] = stack
    return stack


@triton.jit
def _padded_rows(tokens, G: tl.constexpr, SMALL_M: tl.constexpr, LARGE_M: tl.constexpr):
    """Rows the kernel computes for a task of `tokens` tokens."""
    rows = tokens * G
    return tl.where(rows <= SMALL_M, SMALL_M, tl.cdiv(rows, LARGE_M) * LARGE_M)


@triton.jit(do_not_specialize=["num_reqs", "capacity"])
def _enumerate_kernel(qsl_ptr, seq_lens_ptr, block_ptr, d_sum_ptr, tasks_ptr, cost_ptr, count_ptr,
                      num_reqs, capacity, task_stride, cost_stride,
                      G: tl.constexpr, HKV: tl.constexpr, HKV_P: tl.constexpr, BLOCK_R: tl.constexpr,
                      SMALL_M: tl.constexpr, LARGE_M: tl.constexpr, SPLIT_DECODE: tl.constexpr,
                      WITH_COST: tl.constexpr):
    """Tasks of one layer in natural order (request, KV head, block).

    With SPLIT_DECODE, tasks of single-token requests go to list 1 and the others to list 0 (lists are
    `capacity` entries apart); otherwise all go to list 0. count_ptr[layer] receives (count_0, count_1),
    or (0, count) for one list.
    """
    layer = tl.program_id(0)
    FLAT: tl.constexpr = BLOCK_R * HKV_P
    heads = tl.arange(0, HKV_P)
    head_ok = heads < HKV
    block = tl.load(block_ptr + layer * HKV + heads, mask=head_ok, other=1)
    d_sum = tl.load(d_sum_ptr + layer * HKV + heads, mask=head_ok, other=0.0)
    pf_tasks = tasks_ptr + layer.to(tl.int64) * task_stride
    pf_cost = cost_ptr + layer.to(tl.int64) * cost_stride
    dc_tasks = pf_tasks + capacity * 4
    dc_cost = pf_cost + capacity
    n_pf = 0
    n_dc = 0
    for r0 in range(0, num_reqs, BLOCK_R):
        req = r0 + tl.arange(0, BLOCK_R)
        req_ok = req < num_reqs
        lo = tl.load(qsl_ptr + req, mask=req_ok, other=0)
        q_len = tl.load(qsl_ptr + req + 1, mask=req_ok, other=0) - lo
        ctx = tl.load(seq_lens_ptr + req, mask=req_ok, other=0) - q_len
        valid = req_ok[:, None] & head_ok[None, :]
        tokens = tl.reshape(tl.where(valid, q_len[:, None], 0), [FLAT])
        size = tl.reshape(tl.broadcast_to(block[None, :], [BLOCK_R, HKV_P]), [FLAT])
        req_f = tl.reshape(tl.broadcast_to(req[:, None], [BLOCK_R, HKV_P]), [FLAT])
        head_f = tl.reshape(tl.broadcast_to(heads[None, :], [BLOCK_R, HKV_P]), [FLAT])
        ctx_f = tl.reshape(tl.broadcast_to(ctx[:, None], [BLOCK_R, HKV_P]), [FLAT])
        d_sum_f = tl.reshape(tl.broadcast_to(d_sum[None, :], [BLOCK_R, HKV_P]), [FLAT])
        if SPLIT_DECODE:
            decode = tokens == 1
            one = decode.to(tl.int32)
            n_blocks = tl.where(decode, 0, tl.cdiv(tokens, size))
            slot = n_dc + tl.cumsum(one, 0) - one
            act = decode & (slot < capacity)
            ptr = dc_tasks + slot.to(tl.int64) * 4
            tl.store(ptr, req_f, mask=act)
            tl.store(ptr + 1, head_f, mask=act)
            tl.store(ptr + 2, one - 1, mask=act)
            tl.store(ptr + 3, one, mask=act)
            rows = _padded_rows(one, G, SMALL_M, LARGE_M)
            tl.store(dc_cost + slot, rows.to(tl.float32) * (ctx_f + 1).to(tl.float32) * d_sum_f, mask=act)
            n_dc += tl.sum(one, 0)
        else:
            n_blocks = tl.cdiv(tokens, size)
        start = n_pf + tl.cumsum(n_blocks, 0) - n_blocks
        for j in range(0, tl.max(n_blocks, 0)):
            slot = start + j
            act = (j < n_blocks) & (slot < capacity)
            t_begin = j * size
            t_end = tl.minimum(t_begin + size, tokens)
            ptr = pf_tasks + slot.to(tl.int64) * 4
            tl.store(ptr, req_f, mask=act)
            tl.store(ptr + 1, head_f, mask=act)
            tl.store(ptr + 2, t_begin, mask=act)
            tl.store(ptr + 3, t_end, mask=act)
            if WITH_COST:
                rows = _padded_rows(t_end - t_begin, G, SMALL_M, LARGE_M)
                cost = rows.to(tl.float32) * (ctx_f + t_end).to(tl.float32) * d_sum_f
                tl.store(pf_cost + slot, cost, mask=act)
        n_pf += tl.sum(n_blocks, 0)
    n_pf = tl.minimum(n_pf, capacity)
    n_dc = tl.minimum(n_dc, capacity)
    if SPLIT_DECODE:
        tl.store(count_ptr + layer * 2, n_pf)
        tl.store(count_ptr + layer * 2 + 1, n_dc)
    else:
        tl.store(count_ptr + layer * 2, 0)
        tl.store(count_ptr + layer * 2 + 1, n_pf)
    if WITH_COST:  # unused slots sort behind every task
        for s0 in range(n_pf, capacity, FLAT):
            offs = s0 + tl.arange(0, FLAT)
            tl.store(pf_cost + offs, tl.full([FLAT], -1.0, tl.float32), mask=offs < capacity)
        if SPLIT_DECODE:
            for s0 in range(n_dc, capacity, FLAT):
                offs = s0 + tl.arange(0, FLAT)
                tl.store(dc_cost + offs, tl.full([FLAT], -1.0, tl.float32), mask=offs < capacity)


@triton.jit
def _keys(load, rank, IDX_MASK: tl.constexpr):
    """Order-preserving int32 keys of non-negative loads with the worker's tie-break rank in the low bits."""
    return (load.to(tl.int32, bitcast=True) & ~IDX_MASK) | rank


@triton.jit
def _ranks(workers, num_workers, REVERSE: tl.constexpr):
    """Tie-break rank of every worker: its index, or with REVERSE its index counted from the last worker."""
    if REVERSE:
        return tl.where(workers < num_workers, num_workers - 1 - workers, workers)
    return workers


@triton.jit
def _first_round(cost_ptr, worker_ptr, n, num_workers, workers, rank, idle):
    """Tasks 0 .. min(n, num_workers) - 1 go to the idle workers in rank order (task i to rank i)."""
    first = tl.minimum(n, num_workers)
    take = (workers < num_workers) & (rank < first)
    tl.store(worker_ptr + rank, workers, mask=take)
    return idle + tl.load(cost_ptr + rank, mask=take, other=0.0), first


@triton.jit
def _greedy(cost_ptr, worker_ptr, start, end, load, rank, num_workers, IDX_MASK: tl.constexpr,
            REVERSE: tl.constexpr):
    """Assign tasks start .. end - 1 (sorted by cost) one by one to the least-loaded worker.

    Among equally loaded workers the one of lowest rank wins.
    """
    keys = _keys(load, rank, IDX_MASK)
    for i in range(start, end):
        c = tl.load(cost_ptr + i)
        r = tl.min(keys, 0) & IDX_MASK
        hit = rank == r
        load = tl.where(hit, load + c, load)
        keys = tl.where(hit, _keys(load, rank, IDX_MASK), keys)
        w = r
        if REVERSE:
            w = num_workers - 1 - r
        tl.store(worker_ptr + i, w)
    return load


@triton.jit
def _count_by_worker(worker_ptr, n, count, workers, CHUNK: tl.constexpr):
    """Add the number of slots 0 .. n - 1 assigned to each worker to `count`."""
    for s0 in range(0, n, CHUNK):
        offs = s0 + tl.arange(0, CHUNK)
        w = tl.load(worker_ptr + offs, mask=offs < n, other=-1)
        count += tl.sum((w[:, None] == workers[None, :]).to(tl.int32), 0)
    return count


@triton.jit(do_not_specialize=["capacity", "out_capacity", "num_workers"])
def _assign_kernel(cost_ptr, order_ptr, tasks_ptr, count_ptr, worker_ptr, piece_ptr, out_tasks_ptr,
                   out_bounds_ptr, qsl_ptr, seq_lens_ptr, capacity, out_capacity, num_workers, out_task_stride,
                   out_bound_stride,
                   G: tl.constexpr, SMALL_M: tl.constexpr, LARGE_M: tl.constexpr, UNIT: tl.constexpr,
                   W_P: tl.constexpr, CHUNK: tl.constexpr):
    """Greedy assignment of one layer's sorted prefill and decode tasks; per-worker task lists (CSR)."""
    layer = tl.program_id(0).to(tl.int64)
    pf_cost = cost_ptr + layer * 2 * capacity
    dc_cost = pf_cost + capacity
    pf_order = order_ptr + layer * 2 * capacity
    dc_order = pf_order + capacity
    pf_tasks = tasks_ptr + layer * 2 * capacity * 4
    dc_tasks = pf_tasks + capacity * 4
    pf_worker = worker_ptr + layer * (out_capacity + capacity)
    dc_worker = pf_worker + out_capacity
    piece_ptr += layer * out_capacity * 4
    out_tasks_ptr += layer * out_task_stride
    out_bounds_ptr += layer * out_bound_stride
    n_pf = tl.load(count_ptr + layer * 2)
    n_dc = tl.load(count_ptr + layer * 2 + 1)
    workers = tl.arange(0, W_P)
    idle = tl.where(workers < num_workers, 0.0, float("inf"))
    IDX_MASK: tl.constexpr = W_P - 1

    # prefill, complete rounds: longest task first, each to the least-loaded worker
    n_full = n_pf - n_pf % num_workers
    rank = _ranks(workers, num_workers, False)
    load, first = _first_round(pf_cost, pf_worker, n_full, num_workers, workers, rank, idle)
    load = _greedy(pf_cost, pf_worker, first, n_full, load, rank, num_workers, IDX_MASK, False)
    # prefill, last round: cut the remaining tasks into blocks of about wl = remaining work / num_workers
    remaining = 0.0
    for s0 in range(n_full, n_pf, CHUNK):
        offs = s0 + tl.arange(0, CHUNK)
        remaining += tl.sum(tl.load(pf_cost + offs, mask=offs < n_pf, other=0.0), 0)
    wl = remaining / num_workers
    keys = _keys(load, workers, IDX_MASK)
    k = n_full
    for i in range(n_full, n_pf):
        c = tl.load(pf_cost + i)
        src = pf_tasks + tl.load(pf_order + i) * 4
        req = tl.load(src)
        head = tl.load(src + 1)
        t_begin = tl.load(src + 2)
        t_end = tl.load(src + 3)
        tokens = t_end - t_begin
        q_start = tl.load(qsl_ptr + req)
        ctx = tl.load(seq_lens_ptr + req) - (tl.load(qsl_ptr + req + 1) - q_start)
        per_row = c / _padded_rows(tokens, G, SMALL_M, LARGE_M).to(tl.float32)
        per_key_row = per_row / (ctx + t_end).to(tl.float32)  # d_sum of the head
        want = tl.where(per_row > 0, wl / tl.maximum(per_row, 1e-30) / G, tokens.to(tl.float32))
        want = tl.minimum(want, tokens.to(tl.float32))
        piece = tl.maximum(tl.cdiv(tl.ceil(want).to(tl.int32), UNIT) * UNIT, UNIT)
        for p_begin in range(t_begin, t_end, piece):
            p_end = tl.minimum(p_begin + piece, t_end)
            p_cost = _padded_rows(p_end - p_begin, G, SMALL_M, LARGE_M).to(tl.float32) * (ctx + p_end).to(tl.float32)
            w = tl.min(keys, 0) & IDX_MASK
            hit = workers == w
            load = tl.where(hit, load + p_cost * per_key_row, load)
            keys = tl.where(hit, _keys(load, workers, IDX_MASK), keys)
            tl.store(pf_worker + k, w)
            tl.store(piece_ptr + k * 4, req)
            tl.store(piece_ptr + k * 4 + 1, head)
            tl.store(piece_ptr + k * 4 + 2, p_begin)
            tl.store(piece_ptr + k * 4 + 3, p_end)
            k += 1
    n_pf_out = k
    # decode: longest task first, each to the least-loaded worker (LPT); ties go to the highest worker
    # index, so that decode tasks fill the workers from the end opposite to the prefill tie-breaking
    rank = _ranks(workers, num_workers, True)
    load, first = _first_round(dc_cost, dc_worker, n_dc, num_workers, workers, rank, idle)
    load = _greedy(dc_cost, dc_worker, first, n_dc, load, rank, num_workers, IDX_MASK, True)
    tl.debug_barrier()

    # per-worker lists: prefill slots then decode slots, each in slot order (counting sort by worker)
    zero = tl.zeros([W_P], tl.int32)
    count = _count_by_worker(pf_worker, n_pf_out, zero, workers, CHUNK)
    count = _count_by_worker(dc_worker, n_dc, count, workers, CHUNK)
    nxt = tl.cumsum(count, 0) - count
    tl.store(out_bounds_ptr + workers, nxt, mask=workers < num_workers)
    tl.store(out_bounds_ptr + num_workers, n_pf_out + n_dc)
    for s0 in range(0, n_pf_out, CHUNK):
        offs = s0 + tl.arange(0, CHUNK)
        ok = offs < n_pf_out
        w = tl.load(pf_worker + offs, mask=ok, other=-1)
        hit = (w[:, None] == workers[None, :]).to(tl.int32)
        pos = tl.sum(hit * (tl.cumsum(hit, 0) - hit + nxt[None, :]), 1)
        nxt += tl.sum(hit, 0)
        sorted_slot = ok & (offs < n_full)
        piece_slot = ok & (offs >= n_full)
        src = tl.load(pf_order + offs, mask=sorted_slot, other=0)
        for f in tl.static_range(4):
            a = tl.load(pf_tasks + src * 4 + f, mask=sorted_slot, other=0)
            b = tl.load(piece_ptr + offs * 4 + f, mask=piece_slot, other=0)
            tl.store(out_tasks_ptr + pos * 4 + f, tl.where(sorted_slot, a, b), mask=ok)
    for s0 in range(0, n_dc, CHUNK):
        offs = s0 + tl.arange(0, CHUNK)
        ok = offs < n_dc
        w = tl.load(dc_worker + offs, mask=ok, other=-1)
        hit = (w[:, None] == workers[None, :]).to(tl.int32)
        pos = tl.sum(hit * (tl.cumsum(hit, 0) - hit + nxt[None, :]), 1)
        nxt += tl.sum(hit, 0)
        src = tl.load(dc_order + offs, mask=ok, other=0)
        for f in tl.static_range(4):
            tl.store(out_tasks_ptr + pos * 4 + f, tl.load(dc_tasks + src * 4 + f, mask=ok, other=0), mask=ok)


def _batch_shape(query_start_loc: torch.Tensor, num_tokens: int | None,
                 max_query_len: int | None) -> tuple[int, int]:
    if num_tokens is None or max_query_len is None:
        qsl = query_start_loc.cpu()
        if num_tokens is None:
            num_tokens = int(qsl[-1] - qsl[0]) if qsl.numel() > 1 else 0
        if max_query_len is None:
            max_query_len = int((qsl[1:] - qsl[:-1]).max()) if qsl.numel() > 1 else 0
    return int(num_tokens), int(max_query_len)


def _new(workspace: ScheduleWorkspace | None, name: str, shape: tuple[int, ...], dtype: torch.dtype,
         device: torch.device) -> torch.Tensor:
    if workspace is None:
        return torch.empty(shape, dtype=dtype, device=device)
    return workspace.buffer(name, shape, dtype, device)


def build_schedules(layouts: Sequence[LayerLayout], query_start_loc: torch.Tensor, seq_lens: torch.Tensor,
                    num_heads: int, balance: bool = True, *, num_tokens: int | None = None,
                    max_query_len: int | None = None, num_workers: int | None = None,
                    workspace: ScheduleWorkspace | None = None) -> list[Schedule]:
    """Schedules of all `layouts` (one per layer) for one batch, built in a few GPU launches.

    query_start_loc [B + 1] and seq_lens [B] are the int32 GPU tensors later passed to paged_attention.
    num_tokens (an upper bound on query_start_loc[-1] - query_start_loc[0]) and max_query_len are host
    values; when they are not given they are read from query_start_loc, which synchronizes the device.
    The layouts describe the K and V widths seen by attention; their per-layer tables are cached by the
    identity of the layout objects, so passing the same objects every step costs no device work.
    num_workers sets the number of persistent programs (default: WORKERS_PER_SM per SM). With a
    `workspace`, the output buffers are reused across calls.
    """
    if not layouts:
        return []
    device = query_start_loc.device
    if query_start_loc.dtype != torch.int32 or seq_lens.dtype != torch.int32:
        raise ValueError("query_start_loc and seq_lens must be int32")
    hkv = tile_info(layouts[0].k).num_heads
    if num_heads % hkv != 0:
        raise ValueError(f"num_heads={num_heads} is not a multiple of the {hkv} KV heads")
    g = num_heads // hkv
    stack = _layer_stack(layouts, g, device)
    num_layers = len(layouts)
    num_reqs = query_start_loc.numel() - 1
    num_tokens, max_query_len = _batch_shape(query_start_loc, num_tokens, max_query_len)
    large_m = large_tile_rows(g)
    has_small = g <= SMALL_M
    has_large = max_query_len > 1 or g > SMALL_M
    hkv_p = triton.next_power_of_2(hkv)
    meta = dict(G=g, HKV=hkv, HKV_P=hkv_p, BLOCK_R=max(1, 256 // hkv_p), SMALL_M=SMALL_M, LARGE_M=large_m,
                num_warps=4)

    if not balance:
        # blocks hold ROW_BLOCK tokens except the last one of each request and head
        capacity = max(hkv * (num_tokens // ROW_BLOCK + num_reqs) if max_query_len > 1 else hkv * num_reqs, 1)
        tasks = _new(workspace, "natural_tasks", (1, capacity, 4), torch.int32, device)
        bounds = _new(workspace, "natural_bounds", (1, 2), torch.int32, device)
        _enumerate_kernel[(1,)](query_start_loc, seq_lens, stack.fixed_blocks, stack.d_sum, tasks, tasks, bounds,
                                num_reqs, capacity, tasks.stride(0), 0, SPLIT_DECODE=False, WITH_COST=False, **meta)
        schedule = Schedule(tasks, bounds, 0, capacity, False, g, has_small, has_large)
        return [schedule] * num_layers

    workers = num_workers or num_sms(device) * WORKERS_PER_SM
    if not 1 <= workers <= MAX_WORKERS:
        raise ValueError(f"num_workers={workers} is not supported (1 to {MAX_WORKERS})")
    w_p = triton.next_power_of_2(workers)
    prefill_cap = hkv * (num_tokens // stack.min_block + num_reqs) if max_query_len > 1 else 0
    capacity = max(prefill_cap, hkv * num_reqs, 1)
    tasks = torch.empty((num_layers, 2, capacity, 4), dtype=torch.int32, device=device)
    cost = torch.empty((num_layers, 2, capacity), dtype=torch.float32, device=device)
    counts = torch.empty((num_layers, 2), dtype=torch.int32, device=device)
    _enumerate_kernel[(num_layers,)](query_start_loc, seq_lens, stack.row_blocks, stack.d_sum, tasks, cost, counts,
                                     num_reqs, capacity, 2 * capacity * 4, 2 * capacity,
                                     SPLIT_DECODE=True, WITH_COST=True, **meta)
    cost, order = torch.sort(cost, dim=2, descending=True, stable=True)
    out_capacity = capacity + 2 * workers  # prefill tasks plus the blocks of the re-split last round
    worker_of = torch.empty((num_layers, out_capacity + capacity), dtype=torch.int32, device=device)
    pieces = torch.empty((num_layers, out_capacity, 4), dtype=torch.int32, device=device)
    out_tasks = _new(workspace, "tasks", (num_layers, out_capacity + capacity, 4), torch.int32, device)
    bounds = _new(workspace, "bounds", (num_layers, workers + 1), torch.int32, device)
    _assign_kernel[(num_layers,)](cost, order, tasks, counts, worker_of, pieces, out_tasks, bounds,
                                  query_start_loc, seq_lens, capacity, out_capacity, workers, out_tasks.stride(0),
                                  bounds.stride(0),
                                  G=g, SMALL_M=SMALL_M, LARGE_M=large_m, UNIT=tile_tokens(g), W_P=w_p, CHUNK=32,
                                  num_warps=1)
    return [Schedule(out_tasks, bounds, layer, workers, True, g, has_small, has_large)
            for layer in range(num_layers)]
