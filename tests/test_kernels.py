"""Tests of the Triton kernels (fastdc.kernels) against the PyTorch reference (fastdc.kernels.reference)."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np
import pytest
import torch

from fastdc.kernels import Schedule, build_schedules, paged_attention, reference, write_kv
from fastdc.kernels import schedule as sched_mod
from fastdc.kernels.layout import LayerLayout, full_layer_layout, layer_layout

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")

DEVICE = "cuda"
TOLERANCE = {torch.bfloat16: dict(atol=3e-2, rtol=2e-2), torch.float16: dict(atol=4e-3, rtol=4e-3)}
RELATIVE_ERROR = {torch.bfloat16: 1e-2, torch.float16: 2e-3}  # bound on |out - expected| / |expected|


def random_widths(rng: random.Random, num_heads: int, head_dim: int) -> list[int]:
    """Random per-head widths that always include width 1 and the full width."""
    widths = [rng.randint(1, head_dim) for _ in range(num_heads)]
    widths[0] = 1
    widths[-1] = head_dim
    rng.shuffle(widths)
    return widths


def allocate_pages(seq_lens: list[int], layout: LayerLayout, dtype: torch.dtype, seed: int, spare: int = 3):
    """Zeroed pages and a block table that places every request's blocks at random physical blocks."""
    bs = layout.block_size
    blocks = [math.ceil(s / bs) for s in seq_lens]
    total = sum(blocks) + spare
    perm = torch.randperm(total, generator=torch.Generator().manual_seed(seed)).to(torch.int32)
    table = torch.zeros(len(seq_lens), max(blocks + [1]), dtype=torch.int32)
    start = 0
    for b, n in enumerate(blocks):
        table[b, :n] = perm[start:start + n]
        start += n
    pages = torch.zeros(total, layout.page_elems, dtype=dtype, device=DEVICE)
    return pages, table.to(DEVICE)


def slots_of(table: torch.Tensor, seq_lens: list[int], block_size: int) -> torch.Tensor:
    parts = []
    for b, s in enumerate(seq_lens):
        pos = torch.arange(s, device=DEVICE)
        parts.append(table[b, pos // block_size].long() * block_size + pos % block_size)
    return torch.cat(parts) if parts else torch.zeros(0, dtype=torch.long, device=DEVICE)


def fill_cache(pages: torch.Tensor, layout: LayerLayout, table: torch.Tensor, seq_lens: list[int], seed: int):
    """Random K and V for every position of every request, written with the reference."""
    slots = slots_of(table, seq_lens, layout.block_size)
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    hkv = layout.k.num_heads
    k = torch.randn(len(slots), hkv, int(layout.k.dims.max()), generator=gen, device=DEVICE).to(pages.dtype)
    v = torch.randn(len(slots), hkv, int(layout.v.dims.max()), generator=gen, device=DEVICE).to(pages.dtype)
    reference.write_kv(pages, layout, k, v, slots)


@dataclass
class Batch:
    layout: LayerLayout
    pages: torch.Tensor
    table: torch.Tensor
    q: torch.Tensor
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    out_width: int


def make_batch(q_lens: list[int], ctx_lens: list[int], k_dims: list[int], v_dims: list[int], group: int,
               block_size: int, dtype: torch.dtype = torch.bfloat16, seed: int = 0, pad_tokens: int = 0,
               out_extra: int = 0) -> Batch:
    layout = layer_layout(k_dims, v_dims, block_size, DEVICE)
    seq_lens = [c + q for c, q in zip(ctx_lens, q_lens)]
    pages, table = allocate_pages(seq_lens, layout, dtype, seed)
    fill_cache(pages, layout, table, seq_lens, seed)
    num_tokens = sum(q_lens) + pad_tokens
    heads = len(k_dims) * group
    gen = torch.Generator(device=DEVICE).manual_seed(seed + 1)
    q = torch.randn(num_tokens, heads, max(k_dims), generator=gen, device=DEVICE).to(dtype)
    for h in range(heads):
        q[:, h, k_dims[h // group]:] = 0
    qsl = torch.tensor([0] + np.cumsum(q_lens).tolist(), dtype=torch.int32, device=DEVICE)
    return Batch(layout, pages, table, q, qsl, torch.tensor(seq_lens, dtype=torch.int32, device=DEVICE),
                 max(v_dims) + out_extra)


def run_both(batch: Batch, balance: bool, k_pages=None, k_tiles=None, k_table=None, scale: float = 0.125):
    k_pages = batch.pages if k_pages is None else k_pages
    k_tiles = batch.layout.k if k_tiles is None else k_tiles
    shape = (batch.q.shape[0], batch.q.shape[1], batch.out_width)
    expected = torch.empty(shape, dtype=batch.q.dtype, device=DEVICE)
    reference.paged_attention(batch.q, k_pages, k_tiles, batch.pages, batch.layout.v, batch.table,
                              batch.query_start_loc, batch.seq_lens, scale, expected, k_block_table=k_table)
    out = torch.full(shape, float("nan"), dtype=batch.q.dtype, device=DEVICE)
    paged_attention(batch.q, k_pages, k_tiles, batch.pages, batch.layout.v, batch.table, batch.query_start_loc,
                    batch.seq_lens, scale, out, k_block_table=k_table, balance=balance)
    return out, expected


def assert_matches(out: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(out, expected, **TOLERANCE[expected.dtype])
    error = (out.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-30)
    assert error < RELATIVE_ERROR[expected.dtype], f"relative error {error:.3g}"


def check(batch: Batch, balance: bool, **kwargs) -> None:
    assert_matches(*run_both(batch, balance, **kwargs))


# ---------------------------------------------------------------------------------------------------- write_kv


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("block_size", [16, 32, 64])
@pytest.mark.parametrize("head_dim", [64, 128])
def test_write_kv_matches_reference(dtype, block_size, head_dim):
    rng = random.Random(block_size + head_dim)
    hkv = 4
    layout = layer_layout(random_widths(rng, hkv, head_dim), random_widths(rng, hkv, head_dim), block_size, DEVICE)
    num_blocks, tokens = 9, 70
    slots = torch.randperm(num_blocks * block_size, generator=torch.Generator().manual_seed(1))[:tokens]
    slots[::7] = -1
    slots = slots.to(DEVICE)
    k = torch.randn(tokens, hkv, head_dim + 3, device=DEVICE).to(dtype)
    v = torch.randn(tokens, hkv, head_dim, device=DEVICE).to(dtype)
    expected = torch.zeros(num_blocks, layout.page_elems, dtype=dtype, device=DEVICE)
    reference.write_kv(expected, layout, k, v, slots)
    pages = torch.zeros_like(expected)
    write_kv(pages, layout, k, v, slots)
    assert torch.equal(pages, expected)


def test_write_kv_converts_dtype_and_accepts_views():
    layout = layer_layout([5, 16], [16, 3], 16, DEVICE)
    kv = torch.randn(10, 2, 2, 16, device=DEVICE)  # float32 source, K and V as strided views
    slots = torch.arange(10, device=DEVICE) * 3
    expected = torch.zeros(4, layout.page_elems, dtype=torch.bfloat16, device=DEVICE)
    reference.write_kv(expected, layout, kv[:, :, 0], kv[:, :, 1], slots)
    pages = torch.zeros_like(expected)
    write_kv(pages, layout, kv[:, :, 0], kv[:, :, 1], slots)
    assert torch.equal(pages, expected)


# ----------------------------------------------------------------------------------------------- attention


@pytest.mark.parametrize("balance", [True, False])
@pytest.mark.parametrize("block_size", [16, 32, 64])
@pytest.mark.parametrize("group,head_dim", [(1, 64), (4, 128), (8, 128), (8, 64)])
def test_decode(group, head_dim, block_size, balance):
    rng = random.Random(group * 1000 + head_dim + block_size)
    hkv = 4
    ctx = [rng.randint(0, 3000) for _ in range(6)]
    batch = make_batch([1] * 6, ctx, random_widths(rng, hkv, head_dim), random_widths(rng, hkv, head_dim), group,
                       block_size, seed=group + block_size)
    check(batch, balance)


@pytest.mark.parametrize("balance", [True, False])
@pytest.mark.parametrize("group,head_dim", [(1, 64), (4, 128), (8, 64)])
def test_prefill(group, head_dim, balance):
    rng = random.Random(group + head_dim)
    hkv = 4
    batch = make_batch([700, 33, 1, 260], [0, 0, 0, 0], random_widths(rng, hkv, head_dim),
                       random_widths(rng, hkv, head_dim), group, 16, seed=3)
    check(batch, balance)


@pytest.mark.parametrize("balance", [True, False])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("group", [1, 4, 8])
def test_mixed_chunked_prefill(group, dtype, balance):
    """Prefill chunks on existing context together with decode requests and an empty request."""
    rng = random.Random(group)
    hkv, head_dim = 4, 128
    q_lens = [300, 1, 1, 0, 77, 1, 2]
    ctx = [900, 50, 2500, 10, 0, 1, 130]
    batch = make_batch(q_lens, ctx, random_widths(rng, hkv, head_dim), random_widths(rng, hkv, head_dim), group, 32,
                       dtype=dtype, seed=5, pad_tokens=3, out_extra=24)
    out, expected = run_both(batch, balance)
    assert_matches(out, expected)
    assert torch.count_nonzero(out[-3:]) == 0  # tokens that belong to no request
    assert torch.count_nonzero(out[:, :, int(batch.layout.v.dims.max()):]) == 0  # beyond every V width


@pytest.mark.parametrize("balance", [True, False])
def test_long_context(balance):
    rng = random.Random(7)
    hkv, group, head_dim = 2, 4, 128
    batch = make_batch([1, 256, 1], [20000, 16384, 17000], random_widths(rng, hkv, head_dim),
                       random_widths(rng, hkv, head_dim), group, 16, seed=7)
    check(batch, balance)


@pytest.mark.parametrize("balance", [True, False])
@pytest.mark.parametrize("k_block_size", [16, 64])
def test_separate_k_block_table(k_block_size, balance):
    """Keys from another buffer with its own layout and block table; values from the compressed pages."""
    rng = random.Random(k_block_size)
    hkv, group, head_dim = 4, 4, 64
    k_dims = random_widths(rng, hkv, head_dim)
    batch = make_batch([40, 1, 1], [100, 300, 5], k_dims, random_widths(rng, hkv, head_dim), group, 32, seed=11)
    seq_lens = batch.seq_lens.tolist()
    k_layout = full_layer_layout(hkv, head_dim, k_block_size, DEVICE)
    k_pages, k_table = allocate_pages(seq_lens, k_layout, batch.pages.dtype, seed=12, spare=5)
    fill_cache(k_pages, k_layout, k_table, seq_lens, seed=13)
    check(batch, balance, k_pages=k_pages, k_tiles=k_layout.k, k_table=k_table)


def test_explicit_schedule_and_multi_layer_launch():
    """One build for several layers; every layer's attention uses its own schedule."""
    rng = random.Random(3)
    hkv, group, head_dim = 4, 4, 128
    q_lens, ctx = [120, 1, 1, 9], [64, 700, 31, 0]
    batches = [make_batch(q_lens, ctx, random_widths(rng, hkv, head_dim), random_widths(rng, hkv, head_dim),
                          group, 16, seed=20 + i) for i in range(3)]
    qsl, seq_lens = batches[0].query_start_loc, batches[0].seq_lens
    schedules = build_schedules([b.layout for b in batches], qsl, seq_lens, hkv * group,
                                num_tokens=sum(q_lens), max_query_len=max(q_lens))
    for batch, schedule in zip(batches, schedules):
        expected = torch.empty(batch.q.shape[0], batch.q.shape[1], batch.out_width, dtype=batch.q.dtype,
                               device=DEVICE)
        reference.paged_attention(batch.q, batch.pages, batch.layout.k, batch.pages, batch.layout.v, batch.table,
                                  qsl, seq_lens, 0.1, expected)
        out = torch.empty_like(expected)
        paged_attention(batch.q, batch.pages, batch.layout.k, batch.pages, batch.layout.v, batch.table, qsl,
                        seq_lens, 0.1, out, schedule=schedule)
        assert_matches(out, expected)


def test_invalid_inputs_raise():
    batch = make_batch([3, 1], [4, 5], [16, 8], [8, 16], 2, 16)
    out = torch.empty(batch.q.shape[0], batch.q.shape[1], batch.out_width, dtype=batch.q.dtype, device=DEVICE)
    args = (batch.pages, batch.layout.k, batch.pages, batch.layout.v, batch.table, batch.query_start_loc,
            batch.seq_lens, 0.1)
    with pytest.raises(ValueError):
        paged_attention(batch.q[:, :3], *args, out[:, :3])  # 3 query heads for 2 KV heads
    with pytest.raises(ValueError):
        paged_attention(batch.q[..., :8], *args, out)  # q narrower than the widest K head
    with pytest.raises(ValueError):
        paged_attention(batch.q, *args, out[..., :8])  # out narrower than the widest V head
    with pytest.raises(ValueError):
        paged_attention(batch.q, batch.pages.float(), *args[1:], out)  # 4-byte cache
    with pytest.raises(ValueError):
        build_schedules([batch.layout], batch.query_start_loc.long(), batch.seq_lens, 4)


# ----------------------------------------------------------------------------------------------- schedules


def coverage(schedule: Schedule, q_lens: list[int], hkv: int) -> np.ndarray:
    """How often every (request, KV head, token) appears in the schedule's tasks."""
    count = {b: np.zeros((hkv, n), dtype=np.int64) for b, n in enumerate(q_lens)}
    for req, head, t0, t1 in schedule.layer_tasks():
        count[req][head, t0:t1] += 1
    return np.concatenate([c.reshape(-1) for c in count.values()]) if count else np.zeros(0)


def schedule_inputs(q_lens: list[int], ctx: list[int]):
    qsl = torch.tensor([0] + np.cumsum(q_lens).tolist(), dtype=torch.int32, device=DEVICE)
    seq = torch.tensor([c + q for c, q in zip(ctx, q_lens)], dtype=torch.int32, device=DEVICE)
    return qsl, seq


BATCHES = {
    "decode": ([1] * 37, [17 * i + 3 for i in range(37)]),
    "prefill": ([2000, 300, 65, 3], [0, 0, 0, 0]),
    "mixed": ([1000, 1, 1, 0, 1, 250, 1], [4000, 30, 3000, 5, 0, 120, 9000]),
}


@pytest.mark.parametrize("balance", [True, False])
@pytest.mark.parametrize("kind", list(BATCHES))
@pytest.mark.parametrize("group", [1, 4, 8])
def test_every_task_row_is_covered_once(kind, group, balance):
    q_lens, ctx = BATCHES[kind]
    rng = random.Random(len(kind) + group)
    hkv = 8
    layouts = [layer_layout(random_widths(rng, hkv, 128), random_widths(rng, hkv, 128), 16, DEVICE)
               for _ in range(3)]
    qsl, seq = schedule_inputs(q_lens, ctx)
    for schedule in build_schedules(layouts, qsl, seq, hkv * group, balance=balance, num_workers=13):
        assert (coverage(schedule, q_lens, hkv) == 1).all()


def test_unbalanced_schedule_is_natural_order():
    q_lens, ctx = [130, 1, 64], [0, 9, 3]
    layout = layer_layout([16, 128], [128, 16], 16, DEVICE)
    qsl, seq = schedule_inputs(q_lens, ctx)
    schedule = build_schedules([layout], qsl, seq, 4, balance=False)[0]
    expected = [(b, h, t, min(t + 64, n)) for b, n in enumerate(q_lens) for h in range(2) for t in range(0, n, 64)]
    assert schedule.layer_tasks() == expected
    assert not schedule.persistent


def cpu_lpt(costs: list[float], num_workers: int) -> list[list[int]]:
    """Greedy LPT as the decode phase runs it: loads compared through int32 keys whose low bits hold a
    tie-break rank that prefers the highest worker index. Returns task indices per worker in order."""
    w_p = 1 << (num_workers - 1).bit_length()
    mask = np.int32(w_p - 1)
    lanes = np.arange(w_p, dtype=np.int32)
    rank = np.where(lanes < num_workers, num_workers - 1 - lanes, lanes).astype(np.int32)
    order = sorted(range(len(costs)), key=lambda i: -costs[i])
    load = np.full(w_p, np.inf, dtype=np.float32)
    load[:num_workers] = 0
    lists = [[] for _ in range(num_workers)]
    for i in order:
        keys = (load.view(np.int32) & ~mask) | rank
        w = int(np.argmin(keys))
        load[w] = np.float32(load[w] + np.float32(costs[i]))
        lists[w].append(i)
    return lists


@pytest.mark.parametrize("num_workers", [7, 48, 96])
def test_decode_schedule_is_lpt(num_workers):
    rng = random.Random(num_workers)
    hkv, group, num_reqs = 8, 4, 150
    k_dims, v_dims = random_widths(rng, hkv, 128), random_widths(rng, hkv, 128)
    layout = layer_layout(k_dims, v_dims, 16, DEVICE)
    seq_lens = [rng.randint(1, 20000) for _ in range(num_reqs)]
    qsl, seq = schedule_inputs([1] * num_reqs, [s - 1 for s in seq_lens])
    schedule = build_schedules([layout], qsl, seq, hkv * group, num_workers=num_workers)[0]
    # decode cost: 16 padded rows * s_kv * d_sum, evaluated in float32 like the kernel
    d_sum = [sched_mod.head_cost(k, v) for k, v in zip(k_dims, v_dims)]
    costs = [float(np.float32(16) * np.float32(seq_lens[b]) * np.float32(d_sum[h]))
             for b in range(num_reqs) for h in range(hkv)]
    expected = [[(i // hkv, i % hkv, 0, 1) for i in lst] for lst in cpu_lpt(costs, num_workers)]
    assert schedule.worker_tasks() == expected


def model_loads(schedule: Schedule, layout: LayerLayout, q_lens: list[int], ctx: list[int], group: int):
    """Modeled per-worker prefill and decode loads and task costs of a balanced schedule."""
    k_dims, v_dims = layout.k.dims.tolist(), layout.v.dims.tolist()
    large = sched_mod.large_tile_rows(group)
    loads = {"prefill": [], "decode": []}
    costs = {"prefill": [], "decode": []}
    for tasks in schedule.worker_tasks():
        per = {"prefill": 0.0, "decode": 0.0}
        for b, h, t0, t1 in tasks:
            rows = (t1 - t0) * group
            padded = 16 if rows <= 16 else math.ceil(rows / large) * large
            cost = padded * (ctx[b] + t1) * sched_mod.head_cost(k_dims[h], v_dims[h])
            kind = "decode" if q_lens[b] == 1 else "prefill"
            per[kind] += cost
            costs[kind].append(cost)
        for kind in per:
            loads[kind].append(per[kind])
    return loads, costs


@pytest.mark.parametrize("kind", ["prefill", "mixed"])
@pytest.mark.parametrize("group", [2, 4, 8])
def test_balanced_schedule_bounds(kind, group):
    """Greedy list scheduling: no worker exceeds the least-loaded one by more than its largest task."""
    q_lens, ctx = BATCHES[kind]
    rng = random.Random(group)
    hkv = 8
    layout = layer_layout(random_widths(rng, hkv, 128), random_widths(rng, hkv, 128), 16, DEVICE)
    qsl, seq = schedule_inputs(q_lens, ctx)
    schedule = build_schedules([layout], qsl, seq, hkv * group, num_workers=40)[0]
    loads, costs = model_loads(schedule, layout, q_lens, ctx, group)
    for phase in ("prefill", "decode"):
        if costs[phase]:
            assert max(loads[phase]) - min(loads[phase]) <= max(costs[phase]) * (1 + 1e-4)


def test_prefill_blocks_follow_row_scaling():
    """Before the last round, a head's blocks hold 64 * d_sum_max / d_sum_h tokens in whole row tiles."""
    hkv, group = 4, 4
    k_dims, v_dims = [128, 64, 16, 100], [128, 32, 16, 1]
    layout = layer_layout(k_dims, v_dims, 16, DEVICE)
    blocks = sched_mod.row_block_sizes(k_dims, v_dims, group)
    unit = sched_mod.tile_tokens(group)
    assert blocks[0] == 64 and all(b % unit == 0 for b in blocks)
    assert blocks[2] > blocks[1] > blocks[0]
    q_lens = [4096]
    qsl, seq = schedule_inputs(q_lens, [0])
    num_workers = 16
    schedule = build_schedules([layout], qsl, seq, hkv * group, num_workers=num_workers)[0]
    tasks = schedule.layer_tasks()
    n_tasks = sum(math.ceil(4096 / blocks[h]) for h in range(hkv))
    n_full = n_tasks - n_tasks % num_workers
    sizes = sorted((t1 - t0, h) for _, h, t0, t1 in tasks)
    whole = [s for s, h in sizes if s == blocks[h]]
    assert len(whole) >= n_full - hkv  # all complete-round blocks except at most one tail block per head
    assert all(t0 % unit == 0 for _, _, t0, _ in tasks)


def test_workspace_reuses_buffers():
    rng = random.Random(9)
    layout = layer_layout(random_widths(rng, 4, 64), random_widths(rng, 4, 64), 16, DEVICE)
    workspace = sched_mod.ScheduleWorkspace()
    big = schedule_inputs([500, 1, 1], [0, 10, 20])
    small = schedule_inputs([1, 1], [5, 6])
    first = build_schedules([layout], *big, 8, workspace=workspace)[0]
    second = build_schedules([layout], *small, 8, workspace=workspace)[0]
    assert second.tasks.data_ptr() == first.tasks.data_ptr()
    assert second.bounds.data_ptr() == first.bounds.data_ptr()
    fresh = build_schedules([layout], *small, 8)[0]
    assert second.worker_tasks() == fresh.worker_tasks()


def test_schedule_without_host_values_matches():
    """Host values passed explicitly or read from the device give the same schedule."""
    q_lens, ctx = BATCHES["mixed"]
    rng = random.Random(4)
    layout = layer_layout(random_widths(rng, 8, 128), random_widths(rng, 8, 128), 16, DEVICE)
    qsl, seq = schedule_inputs(q_lens, ctx)
    a = build_schedules([layout], qsl, seq, 32)[0]
    b = build_schedules([layout], qsl, seq, 32, num_tokens=sum(q_lens) + 5, max_query_len=max(q_lens))[0]
    assert a.worker_tasks() == b.worker_tasks()
