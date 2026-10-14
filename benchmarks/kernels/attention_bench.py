"""Paged attention kernel benchmark: full vs. compressed KV widths, balanced vs. unbalanced schedules.

    python benchmarks/kernels/attention_bench.py --rate 0.5 --layers 4 --out results/kernels/synthetic
    python benchmarks/kernels/attention_bench.py --plan plans/qwen3_8b.safetensors --out results/kernels/qwen3_8b
    python benchmarks/kernels/attention_bench.py --calibrate --out results/kernels/calibration

Every case (a decode batch at a context length, the prefill of prompts on an existing context, or a mixed
batch of one prefill chunk and a decode batch) times the attention of every benchmarked layer at that
layer's widths, for
    full         every head at the full head dimension (no compression)
    compressed   the per-head widths of --plan, or synthetic widths whose mean is (1 - rate) * head_dim
each with the balanced schedule (balance=True) and the unbalanced one (balance=False). The attention time
of a case is the sum over layers of the per-layer median (CUDA events; the L2 cache is flushed before
every run). build_schedules is timed for all layers at once: GPU time between CUDA events and host time
per call. With a Palu plan, attention reads full-width (reconstructed) keys and the plan's V widths.
Request lengths vary uniformly by +-spread around the case's context length.
Results are written to <out>.json and <out>.csv.

--calibrate instead times layers whose heads have width w (plus one full-width head) and fits
time = a * sum over heads of (d_qk + d_v + c) for decode and for prefill; c estimates the per-key cost of the
workload model of the schedules (fastdc.kernels.schedule.NON_MATMUL_COST).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import statistics
import time
from dataclasses import dataclass

import torch

from fastdc.kernels import build_schedules, paged_attention
from fastdc.kernels.layout import LayerLayout, layer_layout

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16}


@dataclass(frozen=True)
class Case:
    kind: str  # "decode", "prefill" or "mixed"
    batch: int  # requests (for "mixed": decode requests besides the prefill chunk)
    context: int  # tokens already in the cache per request (mean)
    new_tokens: int  # query tokens per request (for "mixed": tokens of the prefill chunk)
    chunk_context: int = 0  # "mixed": tokens already in the cache before the prefill chunk


@dataclass
class Batch:
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    num_blocks: int
    num_tokens: int
    max_query_len: int
    kv_tokens: int  # sum of sequence lengths


def make_batch(case: Case, spread: float, block_size: int, seed: int, device: torch.device) -> Batch:
    rng = random.Random(seed)
    lo, hi = int(case.context * (1 - spread)), int(case.context * (1 + spread))
    contexts = [rng.randint(lo, hi) if case.context else 0 for _ in range(case.batch)]
    q_lens = [case.new_tokens] * case.batch
    if case.kind == "mixed":
        contexts, q_lens = [case.chunk_context] + contexts, [case.new_tokens] + [1] * case.batch
    seq_lens = [c + q for c, q in zip(contexts, q_lens)]
    blocks = [math.ceil(s / block_size) for s in seq_lens]
    total = sum(blocks)
    perm = torch.randperm(total, generator=torch.Generator().manual_seed(seed)).to(torch.int32)
    table = torch.zeros(len(seq_lens), max(blocks), dtype=torch.int32)
    start = 0
    for b, n in enumerate(blocks):
        table[b, :n] = perm[start:start + n]
        start += n
    qsl = [0]
    for q in q_lens:
        qsl.append(qsl[-1] + q)
    return Batch(torch.tensor(qsl, dtype=torch.int32, device=device),
                 torch.tensor(seq_lens, dtype=torch.int32, device=device), table.to(device), total, qsl[-1],
                 max(q_lens), sum(seq_lens))


def synthetic_widths(num_layers: int, num_kv_heads: int, head_dim: int, rate: float, spread: float,
                     seed: int) -> list[tuple[list[int], list[int]]]:
    """Per-layer (K widths, V widths) drawn uniformly around (1 - rate) * head_dim, mean kept at that value."""
    rng = random.Random(seed)
    mean = (1 - rate) * head_dim
    half = spread * mean

    def draw() -> list[int]:
        widths = [mean + rng.uniform(-half, half) for _ in range(num_kv_heads)]
        shift = mean - sum(widths) / num_kv_heads
        return [min(head_dim, max(1, round(w + shift))) for w in widths]

    return [(draw(), draw()) for _ in range(num_layers)]


def plan_widths(path: str, num_layers: int | None) -> tuple[list[tuple[list[int], list[int]]], dict]:
    from fastdc.plan import CompressionPlan

    plan = CompressionPlan.load(path)
    shape = plan.shape
    layers = range(shape.num_layers if num_layers is None else min(num_layers, shape.num_layers))
    widths = []
    for layer in layers:
        k = plan.k_dims[layer].tolist()
        if plan.method == "palu":  # keys are reconstructed to the full head dimension before attention
            k = [shape.head_dim] * shape.num_kv_heads
        widths.append((k, plan.v_dims[layer].tolist()))
    model = dict(num_kv_heads=shape.num_kv_heads, group_size=shape.group_size, head_dim=shape.head_dim,
                 method=plan.method, kv_rate=plan.kv_rate)
    return widths, model


class Timer:
    """Median CUDA-event time of a callable with the L2 cache flushed before every run."""

    def __init__(self, warmup: int, iters: int, device: torch.device):
        self.warmup, self.iters = warmup, iters
        l2 = torch.cuda.get_device_properties(device).L2_cache_size
        self.flush = torch.empty(2 * l2, dtype=torch.uint8, device=device)

    def __call__(self, fn) -> float:
        for _ in range(self.warmup):
            fn()
        events = []
        for _ in range(self.iters):
            self.flush.zero_()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            events.append((start, end))
        torch.cuda.synchronize()
        return statistics.median(s.elapsed_time(e) for s, e in events)


def time_schedules(layouts: list[LayerLayout], batch: Batch, num_heads: int, balance: bool, timer: Timer):
    """(GPU ms between CUDA events, host ms per call) of building the schedules of all layers."""
    def build():
        return build_schedules(layouts, batch.query_start_loc, batch.seq_lens, num_heads, balance=balance,
                               num_tokens=batch.num_tokens, max_query_len=batch.max_query_len)

    gpu = timer(build)
    torch.cuda.synchronize()
    begin = time.perf_counter()
    for _ in range(timer.iters):
        build()
    host = (time.perf_counter() - begin) / timer.iters * 1e3
    torch.cuda.synchronize()
    return gpu, host


def time_layer(layout: LayerLayout, batch: Batch, num_heads: int, schedules: dict, scale: float,
               dtype: torch.dtype, timer: Timer, seed: int) -> dict[bool, float]:
    """Median attention time of one layer under each schedule in `schedules` (keyed by balance)."""
    device = batch.query_start_loc.device
    gen = torch.Generator(device=device).manual_seed(seed)
    pages = torch.randn(batch.num_blocks, layout.page_elems, generator=gen, device=device).to(dtype)
    k_width, v_width = int(layout.k.dims.max()), int(layout.v.dims.max())
    q = torch.randn(batch.num_tokens, num_heads, k_width, generator=gen, device=device).to(dtype)
    out = torch.empty(batch.num_tokens, num_heads, v_width, dtype=dtype, device=device)
    times = {}
    for balance, schedule in schedules.items():
        def attend(schedule=schedule):
            paged_attention(q, pages, layout.k, pages, layout.v, batch.block_table, batch.query_start_loc,
                            batch.seq_lens, scale, out, schedule=schedule)

        times[balance] = timer(attend)
    return times


def run_case(case: Case, widths: list[tuple[list[int], list[int]]], label: str, args, timer: Timer,
             num_heads: int, head_dim: int, device: torch.device) -> list[dict]:
    batch = make_batch(case, args.spread, args.block_size, args.seed, device)
    layouts = [layer_layout(k, v, args.block_size, device) for k, v in widths]
    schedules, sched_times = {}, {}
    for balance in (True, False):
        schedules[balance] = build_schedules(layouts, batch.query_start_loc, batch.seq_lens, num_heads,
                                             balance=balance, num_tokens=batch.num_tokens,
                                             max_query_len=batch.max_query_len)
        sched_times[balance] = time_schedules(layouts, batch, num_heads, balance, timer)
    per_layer = {True: [], False: []}
    cache = {}
    for index, (layout, (k, v)) in enumerate(zip(layouts, widths)):
        key = (tuple(k), tuple(v))
        if key not in cache:
            cache[key] = time_layer(layout, batch, num_heads, {b: schedules[b][index] for b in (True, False)},
                                    head_dim ** -0.5, DTYPES[args.dtype], timer, args.seed + index)
        for balance in (True, False):
            per_layer[balance].append(cache[key][balance])
    kv_elems = sum(sum(k) + sum(v) for k, v in widths)
    rows = []
    for balance in (True, False):
        attention = sum(per_layer[balance])
        rows.append(dict(
            kind=case.kind, batch=case.batch, context=case.context, new_tokens=case.new_tokens,
            chunk_context=case.chunk_context, widths=label,
            balance=balance, layers=len(widths), attention_ms=round(attention, 4),
            attention_ms_per_layer=round(attention / len(widths), 4),
            schedule_gpu_ms=round(sched_times[balance][0], 4), schedule_host_ms=round(sched_times[balance][1], 4),
            mean_k_width=round(sum(sum(k) for k, _ in widths) / (len(widths) * len(widths[0][0])), 2),
            mean_v_width=round(sum(sum(v) for _, v in widths) / (len(widths) * len(widths[0][1])), 2),
            kv_gbytes=round(batch.kv_tokens * kv_elems * 2 / 1e9, 4),
        ))
    return rows


def calibrate(args, timer: Timer, device: torch.device) -> list[dict]:
    """Per-phase fit of the attention time against the widths of the heads of a layer.

    The calibration layer has --calibrate-heads KV heads: one at the full head dimension (so that the
    kernel runs with tiles of the full width, as for any layer whose widest head is not compressed) and
    the others at width w. With many tasks the time follows the total work,
        time = a * sum over heads of (d_qk + d_v + c),
    which is linear in w; the fit gives a and c.
    """
    hkv, g, head_dim = args.calibrate_heads, args.group_size, args.head_dim
    cases = [Case("decode", args.batch, args.calibrate_context, 1), Case("prefill", 1, 0, args.calibrate_prefill)]
    rows = []
    for case in cases:
        batch = make_batch(case, 0.0, args.block_size, args.seed, device)
        points = []
        for w in range(16, head_dim + 1, 16):
            widths = [head_dim] + [w] * (hkv - 1)
            layout = layer_layout(widths, widths, args.block_size, device)
            schedule = build_schedules([layout], batch.query_start_loc, batch.seq_lens, hkv * g, balance=False,
                                       num_tokens=batch.num_tokens, max_query_len=batch.max_query_len)[0]
            ms = time_layer(layout, batch, hkv * g, {False: schedule}, head_dim ** -0.5, DTYPES[args.dtype], timer,
                            args.seed)[False]
            points.append((w, ms))
            rows.append(dict(kind=case.kind, width=w, attention_ms=round(ms, 4)))
        n = len(points)
        mx = sum(w for w, _ in points) / n
        my = sum(t for _, t in points) / n
        slope = sum((w - mx) * (t - my) for w, t in points) / sum((w - mx) ** 2 for w, _ in points)
        intercept = my - slope * mx
        a = slope / (2 * (hkv - 1))  # ms per head-dimension column of one head
        c = (intercept / a - 2 * head_dim) / hkv
        rows.append(dict(kind=case.kind, ms_per_column=round(a, 7), non_matmul_cost=round(c, 1)))
        print(f"{case.kind}: {a * 1e3:.4f} us per column per head; c = {c:.1f} columns")
    return rows


def write(rows: list[dict], config: dict, out: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out + ".json", "w") as f:
        json.dump(dict(config=config, results=rows), f, indent=1)
    fields = []
    for row in rows:
        fields += [k for k in row if k not in fields]
    with open(out + ".csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print("wrote", out + ".json", "and", out + ".csv")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", default=None, help="CompressionPlan file (per-head widths of every layer)")
    ap.add_argument("--rate", type=float, default=0.5, help="synthetic widths: average compression rate")
    ap.add_argument("--width-spread", type=float, default=0.6, help="synthetic widths: relative half-range")
    ap.add_argument("--layers", type=int, default=None,
                    help="number of synthetic layers (default 4), or the first layers of the plan (default all)")
    ap.add_argument("--num-kv-heads", type=int, default=8)
    ap.add_argument("--group-size", type=int, default=4, help="query heads per KV head")
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--batch", type=int, default=32, help="decode batch size")
    ap.add_argument("--decode-context", type=int, nargs="*", default=[1024, 4096, 16384])
    ap.add_argument("--prefill", type=int, nargs="*", default=[1024, 4096], help="prompt lengths")
    ap.add_argument("--prefill-batch", type=int, default=1)
    ap.add_argument("--prefill-context", type=int, default=0, help="cached tokens before each prompt or chunk")
    ap.add_argument("--mixed", type=int, nargs="*", default=[],
                    help="prefill chunk lengths of mixed batches (one chunk plus --batch decode requests at the "
                         "first --decode-context length)")
    ap.add_argument("--spread", type=float, default=0.5, help="relative spread of request lengths")
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--dtype", choices=list(DTYPES), default="bfloat16")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--calibrate", action="store_true", help="fit the per-key cost of the workload model")
    ap.add_argument("--calibrate-heads", type=int, default=32, help="KV heads of the calibration layer")
    ap.add_argument("--calibrate-context", type=int, default=4096)
    ap.add_argument("--calibrate-prefill", type=int, default=4096)
    ap.add_argument("--out", default="results/kernels/attention")
    args = ap.parse_args()

    device = torch.device("cuda")
    timer = Timer(args.warmup, args.iters, device)
    config = dict(vars(args), device=torch.cuda.get_device_name(device),
                  sm_count=torch.cuda.get_device_properties(device).multi_processor_count)
    if args.calibrate:
        write(calibrate(args, timer, device), config, args.out)
        return

    if args.plan:
        compressed, model = plan_widths(args.plan, args.layers)
        config.update(model)
        hkv, g, head_dim = model["num_kv_heads"], model["group_size"], model["head_dim"]
    else:
        hkv, g, head_dim = args.num_kv_heads, args.group_size, args.head_dim
        compressed = synthetic_widths(args.layers or 4, hkv, head_dim, args.rate, args.width_spread, args.seed)
    full = [([head_dim] * hkv, [head_dim] * hkv)] * len(compressed)
    cases = [Case("decode", args.batch, c, 1) for c in args.decode_context]
    cases += [Case("prefill", args.prefill_batch, args.prefill_context, n) for n in args.prefill]
    if args.mixed:
        context = args.decode_context[0] if args.decode_context else 1024
        cases += [Case("mixed", args.batch, context, n, args.prefill_context) for n in args.mixed]
    rows = []
    for case in cases:
        for label, widths in (("full", full), ("compressed", compressed)):
            for row in run_case(case, widths, label, args, timer, hkv * g, head_dim, device):
                rows.append(row)
                print(f"{row['kind']:7s} B={row['batch']:<3d} ctx={row['context']:<6d} new={row['new_tokens']:<5d} "
                      f"{row['widths']:10s} balance={str(row['balance']):5s} attention {row['attention_ms']:9.3f} ms "
                      f"schedule {row['schedule_gpu_ms']:.3f} ms GPU / {row['schedule_host_ms']:.3f} ms host",
                      flush=True)
    write(rows, config, args.out)


if __name__ == "__main__":
    main()
