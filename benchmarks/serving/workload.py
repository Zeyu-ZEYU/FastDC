"""Request traces for the serving experiments, built from the evaluation workloads (fastdc.tasks).

Each request carries its prompt as token ids and a fixed number of output tokens (generated with
`ignore_eos`), so every method produces exactly the same number of tokens per request.
Default output lengths are the average output lengths of the traces.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass

from fastdc.evaluate import encode, use_chat_template
from fastdc.tasks import load_task

OUTPUT_LENGTHS = {"imdb": 37, "humaneval": 139, "gsm8k": 85, "arxiv": 243, "cocktail": 159, "ruler": 20,
                  "scbench": 370, "pg19": 61500}


@dataclass
class Request:
    task: str
    prompt: list[int]
    max_tokens: int


def build(task_name: str, tokenizer, num_requests: int, split: str = "evaluation", seed: int = 0,
          output_len: int | None = None, max_output_len: int | None = None, max_prompt_len: int | None = None,
          task_args: dict | None = None) -> list[Request]:
    task = load_task(task_name, tokenizer=tokenizer, **(task_args or {}))
    chat = use_chat_template(tokenizer, task, "auto")
    out_len = output_len or OUTPUT_LENGTHS.get(task_name, 128)
    if max_output_len:
        out_len = min(out_len, max_output_len)
    requests = []
    for ex in task.split(split, seed):
        if len(ex.turns) == 1:
            prompts = [encode(tokenizer, ex.turns[0], chat)]
        else:  # one request per turn, carrying the conversation so far (prefix caching reuses its KV)
            prompts, history = [], []
            for turn, ref in zip(ex.turns, ex.references):
                history += tokenizer(turn, add_special_tokens=not history)["input_ids"]
                prompts.append(list(history))
                history += tokenizer(" " + str(ref), add_special_tokens=False)["input_ids"]
        for ids in prompts:
            if max_prompt_len and len(ids) > max_prompt_len:
                ids = ids[-max_prompt_len:]
            requests.append(Request(task_name, ids, out_len))
    rng = random.Random(seed)
    while len(requests) < num_requests and requests:
        requests.append(rng.choice(requests[: len(requests)]))
    return requests[:num_requests]


def save(requests: list[Request], path: str) -> None:
    with open(path, "w") as f:
        for r in requests:
            f.write(json.dumps(asdict(r)) + "\n")


def load(path: str) -> list[Request]:
    with open(path) as f:
        return [Request(**json.loads(line)) for line in f if line.strip()]
