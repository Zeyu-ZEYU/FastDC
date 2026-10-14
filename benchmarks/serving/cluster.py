"""Start and stop a serving cluster: one vLLM instance per GPU group plus the load balancer.

Serving methods are given as NAME or NAME=PLAN[:key=value,...]:
    stock                      vLLM's default attention backend, no compression
    full=plans/full.safetensors          FastDC backend without compression
    fastdc=plans/fastdc.safetensors      FastDC
    palu=plans/palu.safetensors          pre-RoPE low-rank baseline (decompression before attention)
    fastdc_ke=plans/fastdc.safetensors:balance=false   FastDC without the kernel enhancement
Every child process receives SIGTERM if this process dies, so interrupted runs leave no servers behind.
"""

from __future__ import annotations

import ctypes
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

_LIBC = ctypes.CDLL("libc.so.6", use_errno=True)
HERE = Path(__file__).resolve().parent


def _die_with_parent() -> None:
    _LIBC.prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG


def port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        return probe.connect_ex(("127.0.0.1", port)) == 0


@dataclass
class MethodSpec:
    name: str
    plan: str | None = None
    options: dict = field(default_factory=dict)

    @classmethod
    def parse(cls, text: str) -> "MethodSpec":
        name, _, rest = text.partition("=")
        if not rest:
            return cls(name)
        plan, _, opts = rest.partition(":")
        options = {}
        for item in filter(None, opts.split(",")):
            key, _, value = item.partition("=")
            options[key] = {"true": True, "false": False}.get(value.lower(), value)
        return cls(name, plan, options)

    def vllm_args(self, profile_path: str | None = None) -> list[str]:
        if self.plan is None:
            return []
        config = {"plan": str(Path(self.plan).resolve()), **self.options}
        if config.pop("profile", False) and profile_path:
            config["profile"] = profile_path
        return ["--attention-backend", "CUSTOM", "--additional-config", json.dumps({"fastdc": config})]


@dataclass
class ClusterSpec:
    model: str
    method: MethodSpec
    gpu_groups: list[str]  # e.g. ["0", "1"] or ["0,1"] for tensor parallelism 2
    run_dir: Path
    max_model_len: int = 8192
    max_num_seqs: int = 64
    max_num_batched_tokens: int = 8192
    gpu_memory_utilization: float = 0.9
    prefix_caching: bool = False  # off by default: a replayed trace must not hit the KV of an earlier run
    base_port: int = 8100
    router_port: int = 8000
    extra_args: list[str] = field(default_factory=list)


class Cluster:
    def __init__(self, spec: ClusterSpec):
        self.spec = spec
        self.procs: list[subprocess.Popen] = []
        self.logs = []
        spec.run_dir.mkdir(parents=True, exist_ok=True)
        self.ports = [spec.base_port + i for i in range(len(spec.gpu_groups))]

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.spec.router_port}"

    def _spawn(self, cmd: list[str], env: dict, log: Path) -> None:
        handle = open(log, "w")
        self.logs.append(handle)
        self.procs.append(subprocess.Popen(cmd, env=env, stdout=handle, stderr=subprocess.STDOUT,
                                           preexec_fn=_die_with_parent, start_new_session=True))

    def start(self, timeout: float = 900.0) -> None:
        s = self.spec
        busy = [p for p in self.ports + [s.router_port] if port_in_use(p)]
        if busy:
            raise RuntimeError(f"ports already in use: {busy}; stop the processes holding them first")
        for gpus, port in zip(s.gpu_groups, self.ports):
            tp = len(gpus.split(","))
            cmd = [sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", s.model,
                   "--host", "127.0.0.1", "--port", str(port), "--dtype", "bfloat16",
                   "--tensor-parallel-size", str(tp), "--max-model-len", str(s.max_model_len),
                   "--max-num-seqs", str(s.max_num_seqs), "--max-num-batched-tokens", str(s.max_num_batched_tokens),
                   "--gpu-memory-utilization", str(s.gpu_memory_utilization),
                   "--enable-prefix-caching" if s.prefix_caching else "--no-enable-prefix-caching",
                   *s.method.vllm_args(str(s.run_dir / f"profile_{port}.jsonl")), *s.extra_args]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpus, VLLM_PLUGINS="fastdc" if s.method.plan else "")
            env.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")  # same sampler for every method
            self._spawn(cmd, env, s.run_dir / f"vllm_{port}.log")
        backends = ",".join(f"http://127.0.0.1:{p}" for p in self.ports)
        self._spawn([sys.executable, str(HERE / "router.py"), "--port", str(s.router_port), "--backends", backends],
                    dict(os.environ), s.run_dir / "router.log")
        deadline = time.time() + timeout
        for port in self.ports + [s.router_port]:
            path = "/health"
            while True:
                if any(p.poll() is not None for p in self.procs):
                    self.stop()
                    raise RuntimeError(f"a server exited during start-up; see the logs in {s.run_dir}")
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=5) as r:
                        if r.status == 200:
                            break
                except OSError:
                    pass
                if time.time() > deadline:
                    self.stop()
                    raise TimeoutError(f"server on port {port} did not become healthy; see {s.run_dir}")
                time.sleep(2)

    def stop(self) -> None:
        for p in self.procs:
            if p.poll() is None:
                try:
                    os.killpg(p.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        deadline = time.time() + 60
        for p in self.procs:
            try:
                p.wait(timeout=max(1.0, deadline - time.time()))
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
        for handle in self.logs:
            handle.close()
        self.procs, self.logs = [], []

    def __enter__(self):
        for sig in (signal.SIGTERM, signal.SIGHUP):
            signal.signal(sig, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()
        return False
