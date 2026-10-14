"""Accuracy of a compression method on a workload, with the reference engine."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch

from fastdc.engine import Engine
from fastdc.tasks import Task

RAW_TEXT_TASKS = ("scbench", "pg19", "ruler", "lcc", "repobench", "sharegpt")  # completion-style prompts


@dataclass
class EvalResult:
    task: str
    split: str
    num_examples: int
    score: float
    cache_bytes: float  # mean stored KV bytes at the end of a request
    full_cache_bytes: float  # mean uncompressed KV bytes of the same requests
    seconds: float
    per_example: list[float] = field(default_factory=list)

    @property
    def measured_kv_rate(self) -> float:
        return 1.0 - self.cache_bytes / self.full_cache_bytes if self.full_cache_bytes else 0.0

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["measured_kv_rate"] = self.measured_kv_rate
        return d


def use_chat_template(tokenizer, task: Task, chat: str) -> bool:
    if chat == "never" or task.name in RAW_TEXT_TASKS:
        return False
    return getattr(tokenizer, "chat_template", None) is not None


def encode(tokenizer, text: str, chat: bool, first: bool = True) -> list[int]:
    if chat:
        ids = tokenizer.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True,
                                            tokenize=True, enable_thinking=False)
        return list(ids["input_ids"] if isinstance(ids, dict) or hasattr(ids, "keys") else ids)
    return tokenizer(text, add_special_tokens=first)["input_ids"]


def stop_ids(tokenizer) -> list[int]:
    ids = {tokenizer.eos_token_id} if tokenizer.eos_token_id is not None else set()
    for name in ("<|im_end|>", "<|eot_id|>", "<|end|>", "<|endoftext|>"):
        tid = tokenizer.convert_tokens_to_ids(name)
        if isinstance(tid, int) and tid != tokenizer.unk_token_id:
            ids.add(tid)
    return sorted(i for i in ids if i is not None)


@torch.inference_mode()
def evaluate(engine: Engine, tokenizer, task: Task, split: str = "evaluation", limit: int | None = None,
             chat: str = "auto", seed: int = 0, progress: bool = False) -> EvalResult:
    examples = task.split(split, seed)
    if limit is not None:
        examples = examples[:limit]
    chat_on = use_chat_template(tokenizer, task, chat)
    eos = stop_ids(tokenizer)
    scores, cache, full = [], [], []
    start = time.time()
    for i, ex in enumerate(examples):
        if len(ex.turns) == 1:
            out = [engine.generate(encode(tokenizer, ex.turns[0], chat_on), ex.max_new_tokens, eos)]
        else:
            turns = [encode(tokenizer, t, False, first=(j == 0)) for j, t in enumerate(ex.turns)]
            out = engine.generate_turns(turns, ex.max_new_tokens, eos)
        texts = [tokenizer.decode(o, skip_special_tokens=True) for o in out]
        score = sum(task.score(t, r) for t, r in zip(texts, ex.references)) / len(ex.references)
        scores.append(score)
        cache.append(engine.cache_bytes())
        full.append(engine.full_cache_bytes(engine.length))
        if progress:
            print(f"[{task.name}] {i + 1}/{len(examples)} score={score:.3f} running={sum(scores) / len(scores):.4f}",
                  flush=True)
    n = max(1, len(scores))
    return EvalResult(task.name, split, len(scores), sum(scores) / n, sum(cache) / n, sum(full) / n,
                      time.time() - start, scores)
