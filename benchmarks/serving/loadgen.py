"""Open-loop load generator: Poisson arrivals, streaming completions, per-request latency records.

At most `max_concurrency` requests are in flight (the concurrent clients); an arrival that finds all
clients busy waits for the next free one, and its latency counts from its scheduled arrival time.
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from dataclasses import asdict, dataclass, field

import aiohttp

from workload import Request


@dataclass
class Record:
    task: str
    prompt_tokens: int
    max_tokens: int
    arrival: float
    sent: float = 0.0
    first_token: float = 0.0
    finished: float = 0.0
    token_times: list[float] = field(default_factory=list)
    output_tokens: int = 0
    ok: bool = False
    error: str = ""
    wall_offset: float = 0.0  # time.time() - time.perf_counter() when the record was made
    backend: str = ""  # instance that served the request (reported by the load balancer)

    @property
    def ttft(self) -> float:
        return self.first_token - self.arrival

    @property
    def jct(self) -> float:
        return self.finished - self.arrival

    @property
    def tbt(self) -> float:
        gaps = [b - a for a, b in zip(self.token_times, self.token_times[1:])]
        return sum(gaps) / len(gaps) if gaps else 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("token_times")
        d.update(ttft=self.ttft, jct=self.jct, tbt=self.tbt, wall_arrival=self.arrival + self.wall_offset,
                 wall_finished=self.finished + self.wall_offset)
        return d


async def _one(session, url: str, model: str, req: Request, rec: Record) -> None:
    body = {"model": model, "prompt": req.prompt, "max_tokens": req.max_tokens, "temperature": 0.0,
            "ignore_eos": True, "stream": True}
    rec.sent = time.perf_counter()
    try:
        async with session.post(url + "/v1/completions", json=body) as resp:
            rec.backend = resp.headers.get("X-FastDC-Backend", "")
            if resp.status != 200:
                rec.error = f"HTTP {resp.status}: {(await resp.text())[:200]}"
                return
            async for line in resp.content:
                if not line.startswith(b"data: {"):
                    continue
                now = time.perf_counter()
                choices = json.loads(line[6:]).get("choices") or []
                if not choices:
                    continue
                text = choices[0].get("text", "")
                if text or choices[0].get("finish_reason") is None:
                    if not rec.first_token:
                        rec.first_token = now
                    rec.token_times.append(now)
                    rec.output_tokens += 1
            rec.finished = time.perf_counter()
            rec.ok = rec.first_token > 0
    except Exception as exc:  # recorded per request; the run continues
        rec.error = repr(exc)


async def run(url: str, model: str, requests: list[Request], rate: float, max_concurrency: int = 16,
              seed: int = 0) -> tuple[list[Record], float]:
    """rate: requests per second (inf = all at once). Returns the records and the makespan in seconds."""
    rng = random.Random(seed)
    slots = asyncio.Semaphore(max_concurrency)
    timeout = aiohttp.ClientTimeout(total=None, sock_read=None)
    records: list[Record] = []
    async with aiohttp.ClientSession(timeout=timeout, connector=aiohttp.TCPConnector(limit=0)) as session:
        start = time.perf_counter()
        tasks = []
        t = start
        for req in requests:
            if rate != float("inf"):
                t += rng.expovariate(rate)
                delay = t - time.perf_counter()
                if delay > 0:
                    await asyncio.sleep(delay)
            rec = Record(req.task, len(req.prompt), req.max_tokens, arrival=max(t, start),
                         wall_offset=time.time() - time.perf_counter())
            records.append(rec)

            async def go(req=req, rec=rec):
                async with slots:
                    await _one(session, url, model, req, rec)

            tasks.append(asyncio.create_task(go()))
        await asyncio.gather(*tasks)
        makespan = time.perf_counter() - start
    return records, makespan
