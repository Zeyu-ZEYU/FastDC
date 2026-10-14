"""Aggregation of per-request records into the reported serving metrics."""

from __future__ import annotations


def _pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    v = sorted(values)
    return v[min(len(v) - 1, int(round(q * (len(v) - 1))))]


def summarize(records, makespan: float) -> dict:
    ok = [r for r in records if r.ok]
    out = dict(requests=len(records), completed=len(ok), failed=len(records) - len(ok), makespan_s=makespan)
    for name in ("ttft", "tbt", "jct"):
        vals = [getattr(r, name) for r in ok]
        out[f"avg_{name}_s"] = sum(vals) / len(vals) if vals else 0.0
        out[f"p99_{name}_s"] = _pct(vals, 0.99)
    tokens = sum(r.output_tokens for r in ok)
    out["throughput_req_s"] = len(ok) / makespan if makespan else 0.0
    out["throughput_tok_s"] = tokens / makespan if makespan else 0.0
    return out
