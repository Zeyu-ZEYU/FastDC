"""Workloads of the evaluation: prompts, references and metrics.

    imdb       IMDb movie genre classification             exact match of the genre
    humaneval  HumanEval code completion                    edit similarity (normalized Levenshtein)
    gsm8k      GSM8K math word problems                     exact match of the final number
    arxiv      arXiv summarization                          ROUGE-2 F1
    cocktail   Cocktail (NQ) information retrieval          exact match of the passage identifier
    ruler      RULER needle-in-a-haystack retrieval         share of needle values found
    scbench    SCBench key-value retrieval, multi-turn      exact match per turn
    pg19       PG-19 book continuation                      ROUGE-2 F1

Held-out workloads:
    mgsm       MGSM (English) math word problems            exact match of the final number
    lcc        LCC Python repository code completion        edit similarity of the next line
    repobench  RepoBench-P (Python, cross-file)             edit similarity of the next line
    bfcl       BFCL simple function calling                 function name and arguments match
    sharegpt   ShareGPT multi-turn dialogue                 ROUGE-2 F1 per turn

Every task splits its examples at random into a calibration subset (20%) and an evaluation subset (80%):
the F_p sweep measures accuracy on the calibration subset and the evaluations use the other one.
"""

from __future__ import annotations

import ast
import io
import json
import random
import re
import uuid
import zipfile
from dataclasses import dataclass, field
from typing import Callable

CALIBRATION_SHARE = 0.2


@dataclass
class Example:
    turns: list[str]  # one prompt per turn; single-turn tasks have one
    references: list  # one reference per turn
    max_new_tokens: int


@dataclass
class Task:
    name: str
    metric: str
    examples: list[Example]
    score: Callable[[str, object], float]
    calibration_texts: list[str] = field(default_factory=list)

    def split(self, which: str, seed: int = 0) -> list[Example]:
        """'calibration' (20%), 'evaluation' (80%) or 'all'."""
        if which == "all":
            return self.examples
        order = list(range(len(self.examples)))
        random.Random(seed).shuffle(order)
        cut = max(1, round(CALIBRATION_SHARE * len(order)))
        chosen = order[:cut] if which == "calibration" else order[cut:]
        return [self.examples[i] for i in sorted(chosen)]


# --------------------------------------------------------------------------- metrics
def _words(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9\- ]", " ", text.lower()).split()


def edit_similarity(a: str, b: str) -> float:
    """1 - Levenshtein distance / length of the longer string."""
    if not a and not b:
        return 1.0
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return 1.0 - previous[-1] / max(len(a), len(b))


def final_number_score(output: str, reference: str) -> float:
    """1 if the number after 'Answer:' (or the last number of the output) equals the reference."""
    found = re.findall(r"Answer:\s*\$?(-?[\d,]*\.?\d+)", output) or re.findall(r"-?[\d,]*\.?\d+", output)
    try:
        return float(bool(found) and abs(float(found[-1].replace(",", "")) - float(reference)) < 1e-6)
    except ValueError:
        return 0.0


def rouge2(output: str, reference: str) -> float:
    from rouge_score import rouge_scorer

    return rouge_scorer.RougeScorer(["rouge2"], use_stemmer=True).score(reference, output)["rouge2"].fmeasure


# --------------------------------------------------------------------------- loaders
def _hf(name, *args, **kwargs):
    from datasets import load_dataset

    return load_dataset(name, *args, **kwargs)


def imdb(limit: int = 200, seed: int = 0, **_) -> Task:
    data = _hf("adrienheymans/imdb-movie-genres", split="test")
    genres = sorted(set(data["genre"]))
    rows = data.shuffle(seed=seed).select(range(min(limit, len(data))))
    options = ", ".join(genres)
    examples = [Example([f"Classify the genre of the movie described below. Choose one of: {options}.\n"
                         f"Answer with the genre only.\n\nTitle: {r['title']}\nPlot: {r['text']}"], [r["genre"]], 16)
                for r in rows]

    def score(output: str, reference: str) -> float:
        words = _words(output)
        return float(bool(words) and words[0] == reference)

    return Task("imdb", "accuracy", examples, score, [e.turns[0] for e in examples])


def humaneval(limit: int = 164, seed: int = 0, **_) -> Task:
    rows = _hf("openai/openai_humaneval", split="test").shuffle(seed=seed)
    rows = rows.select(range(min(limit, len(rows))))
    examples = [Example(["Complete the following Python function. Reply with the code of the function body only.\n\n"
                         + r["prompt"]], [r["canonical_solution"]], 384) for r in rows]

    def score(output: str, reference: str) -> float:
        code = re.sub(r"```(?:python)?", "", output).strip("\n")
        return edit_similarity(code.strip(), reference.strip())

    return Task("humaneval", "edit similarity", examples, score, [e.turns[0] for e in examples])


def gsm8k(limit: int = 300, seed: int = 0, **_) -> Task:
    rows = _hf("openai/gsm8k", "main", split="test").shuffle(seed=seed)
    rows = rows.select(range(min(limit, len(rows))))
    examples = [Example([f"{r['question']}\nSolve the problem step by step. End with a line 'Answer: <number>'."],
                        [r["answer"].split("####")[-1].strip().replace(",", "")], 512) for r in rows]

    return Task("gsm8k", "accuracy", examples, final_number_score, [e.turns[0] for e in examples])


def arxiv(limit: int = 100, seed: int = 0, max_input_chars: int = 24000, **_) -> Task:
    rows = _hf("ccdv/arxiv-summarization", split="test").shuffle(seed=seed)
    rows = rows.select(range(min(limit, len(rows))))
    examples = [Example([f"Write the abstract of the following paper.\n\n{r['article'][:max_input_chars]}\n\nAbstract:"],
                        [r["abstract"]], 384) for r in rows]
    return Task("arxiv", "ROUGE-2", examples, rouge2, [e.turns[0] for e in examples])


def cocktail(limit: int = 100, seed: int = 0, passages: int = 60, **_) -> Task:
    from huggingface_hub import hf_hub_download

    archive = zipfile.ZipFile(hf_hub_download("IR-Cocktail/nq", "nq.zip", repo_type="dataset"))

    def read_jsonl(name):
        with archive.open(name) as handle:
            return [json.loads(line) for line in io.TextIOWrapper(handle, encoding="utf-8")]

    corpora = [{d["_id"]: d for d in read_jsonl(n)} for n in sorted(archive.namelist())
               if n.startswith("corpus/") and n.endswith(".jsonl")]
    queries = {q["_id"]: q["text"] for q in read_jsonl("queries.jsonl")}
    relevant: dict[str, set[str]] = {}
    with archive.open("qrels/test.tsv") as handle:
        for line in list(io.TextIOWrapper(handle, encoding="utf-8"))[1:]:
            qid, did, grade = line.rstrip("\n").split("\t")
            if int(grade) > 0:
                relevant.setdefault(qid, set()).add(did)
    rng = random.Random(seed)
    doc_ids = sorted(corpora[0])
    examples = []
    for qid in rng.sample(sorted(relevant), min(limit, len(relevant))):
        gold = rng.choice(sorted(relevant[qid]))
        chosen = [gold] + rng.sample([d for d in doc_ids if d not in relevant[qid]], passages - 1)
        rng.shuffle(chosen)
        lines = []
        for d in chosen:
            doc = rng.choice(corpora)[d]
            lines.append(f"[{d}] {doc.get('title', '')}: {doc['text']}")
        prompt = ("Below is a list of passages, each starting with its identifier in brackets.\n\n" + "\n".join(lines)
                  + f"\n\nQuestion: {queries[qid]}\nWhich passage answers the question? Reply with its identifier only.")
        examples.append(Example([prompt], [gold], 16))

    def score(output: str, reference: str) -> float:
        found = re.findall(r"doc\d+", output)
        return float(bool(found) and found[0] == reference)

    return Task("cocktail", "accuracy", examples, score, [e.turns[0] for e in examples])


_NOISE = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "


def ruler(limit: int = 100, seed: int = 0, context_tokens: int = 8192, tokenizer=None, **_) -> Task:
    """Needle-in-a-haystack tasks in the style of RULER (single needle, multiple keys, multiple values)."""
    rng = random.Random(seed)
    chars_per_token = 4.0
    if tokenizer is not None:
        chars_per_token = len(_NOISE * 50) / len(tokenizer(_NOISE * 50, add_special_tokens=False)["input_ids"])
    haystack_chars = int(context_tokens * chars_per_token * 0.95)
    examples = []
    for i in range(limit):
        kind = ("single", "multikey", "multivalue")[i % 3]
        sentences = (_NOISE * (haystack_chars // len(_NOISE) + 1))[:haystack_chars].split(". ")
        key = str(uuid.UUID(int=rng.getrandbits(128)))
        values = [str(rng.randrange(10**6, 10**7)) for _ in range(4 if kind == "multivalue" else 1)]
        needles = [f"One of the special magic numbers for {key} is: {v}" for v in values]
        if kind == "multikey":
            for _ in range(3):
                other = str(uuid.UUID(int=rng.getrandbits(128)))
                needles.append(f"One of the special magic numbers for {other} is: {rng.randrange(10**6, 10**7)}")
        for n in needles:
            sentences.insert(rng.randrange(len(sentences)), n)
        context = ". ".join(sentences)
        question = (f"What are all the special magic numbers for {key} mentioned in the provided text?"
                    if kind == "multivalue" else f"What is the special magic number for {key} mentioned in the provided text?")
        prompt = ("Some special magic numbers are hidden within the following text. Make sure to memorize them. "
                  f"I will quiz you about the numbers afterwards.\n{context}\n{question} "
                  "The special magic numbers mentioned in the provided text are")
        examples.append(Example([prompt], [values], 32))

    def score(output: str, reference: list[str]) -> float:
        return sum(v in output for v in reference) / len(reference)

    return Task("ruler", "accuracy", examples, score, [e.turns[0] for e in examples])


def scbench(limit: int = 20, seed: int = 0, context_tokens: int = 8192, turns: int = 5, tokenizer=None, **_) -> Task:
    """SCBench key-value retrieval in multi-turn mode: one shared context, several queries in one conversation.

    The key-value dictionary is subsampled to about `context_tokens` tokens; the queried pairs are always kept.
    """
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    path = hf_hub_download("microsoft/SCBench", "scbench_kv/test-00000-of-00001.parquet", repo_type="dataset")
    rows = pq.read_table(path).to_pylist()
    rng = random.Random(seed)
    rng.shuffle(rows)
    examples = []
    for row in rows[:limit]:
        qa = ast.literal_eval(row["multi_turns"]) if isinstance(row["multi_turns"], str) else row["multi_turns"]
        qa = qa[:turns]
        pairs = json.loads(row["context"][row["context"].index("{"):])
        asked = {re.search(r'"([0-9a-f-]{36})"', t["input"]).group(1) for t in qa}
        others = [k for k in pairs if k not in asked]
        sample = json.dumps(dict(list(pairs.items())[:20]), indent=0)
        per_pair = len(tokenizer(sample, add_special_tokens=False)["input_ids"]) / 20 if tokenizer is not None else 60.0
        budget = max(0, int(context_tokens / per_pair) - len(asked))
        keep = set(rng.sample(others, min(budget, len(others)))) | asked
        sub = {k: v for k, v in pairs.items() if k in keep}
        context = ("Extract the value corresponding to the specified key in the JSON object below.\n\n"
                   + json.dumps(sub, indent=0))
        prompts = [context + "\n" + qa[0]["input"]] + [t["input"] for t in qa[1:]]
        examples.append(Example(prompts, [t["answer"] for t in qa], 48))

    def score(output: str, reference: str) -> float:
        return float(reference in output)

    return Task("scbench", "accuracy", examples, score, [e.turns[0] for e in examples])


def pg19(limit: int = 20, seed: int = 0, prompt_tokens: int = 277, new_tokens: int = 1024, tokenizer=None, **_) -> Task:
    """Book continuation: the first `prompt_tokens` tokens of a PG-19 test book, ROUGE-2 against the true continuation."""
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    if tokenizer is None:
        raise ValueError("pg19 needs the tokenizer to cut prompts and references at token boundaries")
    path = hf_hub_download("emozilla/pg19", "data/test-00000-of-00001-29a571947c0b5ccc.parquet", repo_type="dataset")
    books = pq.read_table(path).column("text").to_pylist()
    random.Random(seed).shuffle(books)
    examples = []
    for text in books[:limit]:
        ids = tokenizer(text[: (prompt_tokens + new_tokens) * 8], add_special_tokens=False)["input_ids"]
        prompt = tokenizer.decode(ids[:prompt_tokens])
        reference = tokenizer.decode(ids[prompt_tokens : prompt_tokens + new_tokens])
        examples.append(Example([prompt], [reference], new_tokens))
    return Task("pg19", "ROUGE-2", examples, rouge2, [e.turns[0] for e in examples])


def _parquet(repo: str, path: str) -> list[dict]:
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    return pq.read_table(hf_hub_download(repo, path, repo_type="dataset")).to_pylist()


def mgsm(limit: int = 250, seed: int = 0, **_) -> Task:
    rows = _parquet("juletxara/mgsm", "en/test-00000-of-00001.parquet")
    random.Random(seed).shuffle(rows)
    examples = [Example([f"{r['question']}\nSolve the problem step by step. End with a line 'Answer: <number>'."],
                        [str(r["answer_number"])], 512) for r in rows[:limit]]
    return Task("mgsm", "accuracy", examples, final_number_score, [e.turns[0] for e in examples])


def _first_line(text: str) -> str:
    lines = [line for line in text.strip("\n").splitlines() if line.strip()]
    return lines[0] if lines else ""


def _next_line_score(output: str, reference: str) -> float:
    return edit_similarity(_first_line(output).strip(), reference.strip())


def lcc(limit: int = 200, seed: int = 0, max_input_chars: int = 24000, **_) -> Task:
    rows = _parquet("microsoft/LCC_python", "data/test-00000-of-00001-88baa4889d1ac8f9.parquet")
    random.Random(seed).shuffle(rows)
    examples = [Example([r["context"][-max_input_chars:]], [r["gt"]], 64) for r in rows[:limit]]
    return Task("lcc", "edit similarity", examples, _next_line_score, [e.turns[0] for e in examples])


def repobench(limit: int = 200, seed: int = 0, max_input_chars: int = 24000, **_) -> Task:
    rows = _parquet("tianyang/repobench_python_v1.1", "data/cross_file_first-00000-of-00002-baebda7f3a6e980a.parquet")
    random.Random(seed).shuffle(rows)
    examples = []
    for r in rows[:limit]:
        snippets = ast.literal_eval(r["context"]) if isinstance(r["context"], str) else r["context"]
        context = "".join(f"# Path: {c['path']}\n{c['snippet']}\n\n" for c in snippets)
        prompt = f"{context}# Path: {r['file_path']}\n{r['import_statement']}\n{r['cropped_code']}\n"
        examples.append(Example([prompt[-max_input_chars:]], [r["next_line"]], 64))
    return Task("repobench", "edit similarity", examples, _next_line_score, [e.turns[0] for e in examples])


def bfcl(limit: int = 400, seed: int = 0, **_) -> Task:
    """BFCL simple category: the model replies with one Python-style call, checked against the possible answers."""
    from huggingface_hub import hf_hub_download

    repo = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"
    read = lambda name: [json.loads(line) for line in open(hf_hub_download(repo, name, repo_type="dataset"))]
    questions = read("BFCL_v3_simple.json")
    answers = {a["id"]: a["ground_truth"][0] for a in read("possible_answer/BFCL_v3_simple.json")}
    random.Random(seed).shuffle(questions)
    examples = []
    for q in questions[:limit]:
        functions = json.dumps(q["function"], indent=1)
        user = q["question"][0][-1]["content"]
        prompt = (f"You can call the following functions:\n{functions}\n\nRequest: {user}\n"
                  "Reply with exactly one function call in Python syntax, e.g. name(arg=value), and nothing else.")
        examples.append(Example([prompt], [answers[q["id"]]], 128))

    def score(output: str, reference: dict) -> float:
        match = re.search(r"[A-Za-z_][\w.]*\(.*\)", output, re.S)
        if not match:
            return 0.0
        try:
            call = ast.parse(match.group(0), mode="eval").body
        except SyntaxError:
            return 0.0
        if not isinstance(call, ast.Call):
            return 0.0
        name = ast.unparse(call.func)
        (ref_name, params), = reference.items()
        if name != ref_name:
            return 0.0
        try:
            given = {kw.arg: ast.literal_eval(kw.value) for kw in call.keywords}
        except ValueError:
            return 0.0
        for param, allowed in params.items():
            if param not in given:
                if "" not in allowed:
                    return 0.0
            elif given[param] not in allowed:
                return 0.0
        return float(all(param in params for param in given))

    return Task("bfcl", "accuracy", examples, score, [e.turns[0] for e in examples])


def sharegpt(limit: int = 50, seed: int = 0, turns: int = 3, max_turn_chars: int = 4000, **_) -> Task:
    """Multi-turn dialogue: user turns are replayed in one conversation; ROUGE-2 against the recorded replies."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download("anon8231489123/ShareGPT_Vicuna_unfiltered", "ShareGPT_V3_unfiltered_cleaned_split.json",
                           repo_type="dataset")
    convs = json.load(open(path))
    rng = random.Random(seed)
    rng.shuffle(convs)
    examples = []
    for conv in convs:
        msgs = conv.get("conversations", [])
        pairs = [(a["value"], b["value"]) for a, b in zip(msgs[::2], msgs[1::2])
                 if a.get("from") == "human" and b.get("from") == "gpt" and a["value"].strip() and b["value"].strip()]
        if len(pairs) < turns:
            continue
        pairs = pairs[:turns]
        prompts = [f"User: {u[:max_turn_chars]}\nAssistant:" for u, _ in pairs]
        examples.append(Example(prompts, [r for _, r in pairs], 256))
        if len(examples) == limit:
            break
    return Task("sharegpt", "ROUGE-2", examples, rouge2, [e.turns[0] for e in examples])


TASKS: dict[str, Callable[..., Task]] = {
    "imdb": imdb, "humaneval": humaneval, "gsm8k": gsm8k, "arxiv": arxiv, "cocktail": cocktail,
    "ruler": ruler, "scbench": scbench, "pg19": pg19,
    "mgsm": mgsm, "lcc": lcc, "repobench": repobench, "bfcl": bfcl, "sharegpt": sharegpt,
}


def load_task(name: str, tokenizer=None, **kwargs) -> Task:
    if name not in TASKS:
        raise ValueError(f"unknown task {name!r}; choose from {sorted(TASKS)}")
    return TASKS[name](tokenizer=tokenizer, **kwargs)
