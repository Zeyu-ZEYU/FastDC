"""Load balancer in front of several vLLM instances (Quart).

Every request goes to the instance with the shortest token-based queue: the sum, over the requests the
instance is serving, of their prompt tokens plus the output tokens they still have to produce.

    python benchmarks/serving/router.py --port 8000 --backends http://127.0.0.1:8100,http://127.0.0.1:8101
"""

from __future__ import annotations

import argparse
import asyncio
import json

import aiohttp
from quart import Quart, Response, request


class Balancer:
    def __init__(self, backends: list[str]):
        self.backends = backends
        self.queued = {b: 0 for b in backends}
        self.lock = asyncio.Lock()

    async def acquire(self, tokens: int) -> str:
        async with self.lock:
            backend = min(self.backends, key=lambda b: self.queued[b])
            self.queued[backend] += tokens
            return backend

    async def release(self, backend: str, tokens: int) -> None:
        async with self.lock:
            self.queued[backend] -= tokens


def create_app(backends: list[str]) -> Quart:
    app = Quart(__name__)
    balancer = Balancer(backends)
    timeout = aiohttp.ClientTimeout(total=None, sock_read=None)
    state: dict = {}

    @app.before_serving
    async def startup():
        state["session"] = aiohttp.ClientSession(timeout=timeout, connector=aiohttp.TCPConnector(limit=0))

    @app.after_serving
    async def shutdown():
        await state["session"].close()

    @app.get("/health")
    async def health():
        return {"backends": balancer.backends, "queued_tokens": balancer.queued}

    @app.post("/v1/completions")
    async def completions():
        body = await request.get_json()
        prompt = body.get("prompt")
        prompt_tokens = len(prompt) if isinstance(prompt, list) else len(str(prompt)) // 4
        pending = prompt_tokens + int(body.get("max_tokens", 16))
        backend = await balancer.acquire(pending)
        session = state["session"]
        headers = {"X-FastDC-Backend": backend}
        if not body.get("stream"):
            try:
                async with session.post(backend + "/v1/completions", json=body) as resp:
                    return Response(await resp.read(), status=resp.status, content_type="application/json",
                                    headers=headers)
            finally:
                await balancer.release(backend, pending)

        async def stream():
            remaining = pending
            try:
                async with session.post(backend + "/v1/completions", json=body) as resp:
                    async for line in resp.content:
                        if line.startswith(b"data: {"):
                            remaining_now = max(prompt_tokens, remaining - 1)
                            await balancer.release(backend, remaining - remaining_now)
                            remaining = remaining_now
                        yield line
            finally:
                await balancer.release(backend, remaining)

        return Response(stream(), content_type="text/event-stream", headers=headers)

    return app


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--backends", required=True, help="comma-separated base URLs of the vLLM instances")
    args = ap.parse_args()
    app = create_app([b.rstrip("/") for b in args.backends.split(",")])
    app.run(host=args.host, port=args.port, use_reloader=False)


if __name__ == "__main__":
    main()
