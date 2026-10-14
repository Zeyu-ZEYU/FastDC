import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks" / "serving"))

pytest.importorskip("quart")

from cluster import MethodSpec  # noqa: E402
from loadgen import Record  # noqa: E402
from metrics import summarize  # noqa: E402
from router import Balancer  # noqa: E402
import workload  # noqa: E402


def test_balancer_picks_shortest_token_queue():
    async def scenario():
        b = Balancer(["a", "b"])
        first = await b.acquire(100)
        second = await b.acquire(10)
        third = await b.acquire(5)
        await b.release(first, 100)
        fourth = await b.acquire(1)
        return first, second, third, fourth

    assert asyncio.run(scenario()) == ("a", "b", "b", "a")


def test_record_latencies_and_summary():
    r = Record("t", 10, 3, arrival=0.0, first_token=0.5, finished=1.0, token_times=[0.5, 0.75, 1.0],
               output_tokens=3, ok=True)
    assert r.ttft == 0.5 and r.jct == 1.0 and r.tbt == pytest.approx(0.25)
    failed = Record("t", 10, 3, arrival=0.0)
    s = summarize([r, failed], makespan=2.0)
    assert s["completed"] == 1 and s["failed"] == 1
    assert s["avg_ttft_s"] == 0.5 and s["throughput_tok_s"] == 1.5


def test_method_spec_parsing():
    assert MethodSpec.parse("stock").vllm_args() == []
    m = MethodSpec.parse("fastdc_ke=plans/p.safetensors:balance=false,attention=reference")
    assert m.name == "fastdc_ke" and m.options == {"balance": False, "attention": "reference"}
    args = m.vllm_args()
    assert args[:2] == ["--attention-backend", "CUSTOM"] and '"balance": false' in args[3]


def test_trace_roundtrip(tmp_path):
    reqs = [workload.Request("t", [1, 2, 3], 5), workload.Request("t", [4], 7)]
    path = str(tmp_path / "trace.jsonl")
    workload.save(reqs, path)
    assert workload.load(path) == reqs
