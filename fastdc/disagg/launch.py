"""Start a disaggregated serving cluster: prefill instances, decode instances (vLLM with the KV connector) and
the proxy.

    python -m fastdc.disagg.launch --model Qwen/Qwen3-8B --plan plans/fastdc.safetensors \\
        --prefill-gpus 0 --decode-gpus 1 --bandwidth-gbps 10

Requests go to the proxy (OpenAI completions API, `http://127.0.0.1:<port-base>`). Without `--plan`, vLLM's
default attention backend runs both roles. The command serves until it is interrupted; `--smoke N` instead
sends N short requests, prints the completions and stops. Every child process receives SIGTERM if this
process dies.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

CONNECTOR = {"kv_connector": "FastDCKVConnector", "kv_connector_module_path": "fastdc.disagg.connector"}
SMOKE_PROMPTS = ["The capital of France is", "List three prime numbers:", "Water boils at",
                 "The largest planet of the solar system is"]
_LIBC = ctypes.CDLL("libc.so.6", use_errno=True)


def _die_with_parent() -> None:
    _LIBC.prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG


def _port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        return probe.connect_ex(("127.0.0.1", port)) == 0


class DisaggCluster:
    def __init__(self, args, run_dir: Path):
        self.args = args
        self.run_dir = run_dir
        run_dir.mkdir(parents=True, exist_ok=True)
        self.procs: list[subprocess.Popen] = []
        self.logs = []
        base = args.port_base
        self.prefill = [(g, base + 100 + i, base + 1100 + i) for i, g in enumerate(args.prefill_gpus.split(";"))]
        self.decode = [(g, base + 200 + i, base + 1200 + i) for i, g in enumerate(args.decode_gpus.split(";"))]
        self.proxy_port = base

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.proxy_port}"

    def _spawn(self, cmd: list[str], env: dict, log: Path) -> None:
        handle = open(log, "w")
        self.logs.append(handle)
        self.procs.append(subprocess.Popen(cmd, env=env, stdout=handle, stderr=subprocess.STDOUT,
                                           preexec_fn=_die_with_parent, start_new_session=True))

    def _serve(self, role: str, gpus: str, port: int, transfer_port: int) -> None:
        a = self.args
        extra = {"transfer_host": "127.0.0.1", "transfer_port": transfer_port,
                 "stats_path": str(self.run_dir / f"{role}_{port}_events.jsonl")}
        if role == "prefill" and a.bandwidth_gbps:
            extra["bandwidth_gbps"] = a.bandwidth_gbps
        transfer = dict(CONNECTOR, kv_role="kv_producer" if role == "prefill" else "kv_consumer",
                        kv_connector_extra_config=extra)
        cmd = [sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", a.model, "--host", "127.0.0.1",
               "--port", str(port), "--dtype", "bfloat16", "--tensor-parallel-size", str(len(gpus.split(","))),
               "--max-model-len", str(a.max_model_len), "--max-num-batched-tokens", str(a.max_model_len),
               "--no-enable-prefix-caching", "--gpu-memory-utilization", str(a.gpu_memory_utilization),
               "--kv-transfer-config", json.dumps(transfer)]
        if a.plan:
            cmd += ["--attention-backend", "CUSTOM",
                    "--additional-config", json.dumps({"fastdc": {"plan": str(Path(a.plan).resolve())}})]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpus, VLLM_PLUGINS="fastdc" if a.plan else "")
        env.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
        self._spawn(cmd, env, self.run_dir / f"{role}_{port}.log")

    def __enter__(self) -> "DisaggCluster":
        ports = [p for _, p, _ in self.prefill + self.decode] + [t for _, _, t in self.prefill + self.decode]
        busy = [p for p in ports + [self.proxy_port] if _port_in_use(p)]
        if busy:
            raise RuntimeError(f"ports already in use: {busy}")
        for gpus, port, tport in self.prefill:
            self._serve("prefill", gpus, port, tport)
        for gpus, port, tport in self.decode:
            self._serve("decode", gpus, port, tport)
        cmd = [sys.executable, "-m", "fastdc.disagg.proxy", "--host", "127.0.0.1", "--port", str(self.proxy_port),
               "--log-path", str(self.run_dir / "proxy.jsonl")]
        cmd += sum((["--prefill", f"http://127.0.0.1:{p}"] for _, p, _ in self.prefill), [])
        cmd += sum((["--decode", f"http://127.0.0.1:{p}"] for _, p, _ in self.decode), [])
        self._spawn(cmd, dict(os.environ), self.run_dir / "proxy.log")
        deadline = time.time() + 900
        for port in [p for _, p, _ in self.prefill + self.decode] + [self.proxy_port]:
            while True:
                if any(p.poll() is not None for p in self.procs):
                    self.__exit__()
                    raise RuntimeError(f"a server exited during start-up; see the logs in {self.run_dir}")
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as r:
                        if r.status == 200:
                            break
                except OSError:
                    pass
                if time.time() > deadline:
                    self.__exit__()
                    raise TimeoutError(f"port {port} did not become healthy; see the logs in {self.run_dir}")
                time.sleep(2)
        return self

    def __exit__(self, *exc) -> bool:
        for p in self.procs:
            if p.poll() is None:
                try:
                    os.killpg(p.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        for p in self.procs:
            try:
                p.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
        for handle in self.logs:
            handle.close()
        self.procs, self.logs = [], []
        return False


def complete(url: str, model: str, prompt: str, max_tokens: int) -> str:
    body = json.dumps({"model": model, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0.0}).encode()
    req = urllib.request.Request(url + "/v1/completions", body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read())["choices"][0]["text"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--plan", default=None, help="compression plan; vLLM's default backend when omitted")
    ap.add_argument("--prefill-gpus", default="0", help="GPU groups of the prefill instances, separated by ';'")
    ap.add_argument("--decode-gpus", default="1", help="GPU groups of the decode instances, separated by ';'")
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    ap.add_argument("--bandwidth-gbps", type=float, default=None, help="sending rate limit of the prefill side")
    ap.add_argument("--port-base", type=int, default=8000, help="proxy port; instances use base+100.., base+200..")
    ap.add_argument("--run-dir", default="results/disagg", help="logs of the instances, proxy and transfers")
    ap.add_argument("--smoke", type=int, default=0, help="send this many short requests, print them and stop")
    args = ap.parse_args()
    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    with DisaggCluster(args, Path(args.run_dir)) as cluster:
        print(f"proxy ready at {cluster.url}", flush=True)
        if args.smoke:
            for prompt in (SMOKE_PROMPTS * args.smoke)[: args.smoke]:
                print(f">>> {prompt!r}\n{complete(cluster.url, args.model, prompt, 24)}", flush=True)
            return
        try:
            while all(p.poll() is None for p in cluster.procs):
                time.sleep(5)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
