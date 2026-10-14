"""Serving experiment: for every method, start the cluster, replay the same trace at each arrival rate,
and write per-request records and summaries.

    python benchmarks/serving/run.py --model Qwen/Qwen3-8B --gpus 0,1 --task arxiv --num-requests 200 \\
        --rates 0.5,1,2 --method palu=plans/palu.safetensors --method fastdc=plans/fastdc.safetensors \\
        --out results/serving/arxiv

Use `--rates inf` to send all requests at once (throughput). Results: <out>/<method>/rate_<r>.json and
<out>/summary.json.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cluster import Cluster, ClusterSpec, MethodSpec  # noqa: E402
from loadgen import run as run_load  # noqa: E402
from metrics import summarize  # noqa: E402
import workload  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--gpus", default="0", help="GPU groups separated by ';', GPUs of one group by ',' (TP)")
    ap.add_argument("--method", action="append", required=True, help="NAME or NAME=PLAN[:key=value,...]")
    ap.add_argument("--task", required=True)
    ap.add_argument("--task-args", default="{}")
    ap.add_argument("--num-requests", type=int, default=100)
    ap.add_argument("--rates", default="1", help="comma-separated requests/s; 'inf' sends everything at once")
    ap.add_argument("--max-concurrency", type=int, default=16)
    ap.add_argument("--output-len", type=int, default=None)
    ap.add_argument("--max-output-len", type=int, default=None)
    ap.add_argument("--max-prompt-len", type=int, default=None)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--max-num-seqs", type=int, default=64)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--out", required=True)
    ap.add_argument("--vllm-arg", action="append", default=[], help="extra argument passed to vllm serve")
    ap.add_argument("--port-base", type=int, default=8000, help="load balancer port; instances use base+100..")
    ap.add_argument("--prefix-caching", action="store_true",
                    help="enable vLLM's prefix caching (multi-turn workloads); every rate then runs on fresh servers")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    requests = workload.build(args.task, tok, args.num_requests, output_len=args.output_len,
                              max_output_len=args.max_output_len, max_prompt_len=args.max_prompt_len,
                              task_args=json.loads(args.task_args))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    workload.save(requests, str(out / "trace.jsonl"))
    rates = [float(r) for r in args.rates.split(",")]
    summary = {}
    for text in args.method:
        method = MethodSpec.parse(text)
        spec = ClusterSpec(args.model, method, args.gpus.split(";"), out / method.name / "logs",
                           max_model_len=args.max_model_len, max_num_seqs=args.max_num_seqs,
                           max_num_batched_tokens=max(args.max_model_len, 2048),
                           gpu_memory_utilization=args.gpu_memory_utilization, extra_args=args.vllm_arg,
                           base_port=args.port_base + 100, router_port=args.port_base)
        spec.prefix_caching = args.prefix_caching
        groups = [[rate] for rate in rates] if args.prefix_caching else [rates]
        for group in groups:
            with Cluster(spec) as cluster:
                warmup = workload.build(args.task, tok, args.warmup, split="calibration",
                                        output_len=8, task_args=json.loads(args.task_args))
                asyncio.run(run_load(cluster.url, args.model, warmup, float("inf"), args.max_concurrency))
                for rate in group:
                    records, makespan = asyncio.run(run_load(cluster.url, args.model, requests, rate,
                                                             args.max_concurrency))
                    result = dict(method=method.name, rate=rate, task=args.task, summary=summarize(records, makespan),
                                  records=[r.to_dict() for r in records])
                    (out / method.name).mkdir(parents=True, exist_ok=True)
                    with open(out / method.name / f"rate_{rate:g}.json", "w") as f:
                        json.dump(result, f, indent=1)
                    summary.setdefault(method.name, {})[f"{rate:g}"] = result["summary"]
                    print(json.dumps({"method": method.name, "rate": rate, **result["summary"]}), flush=True)
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
