"""Offline stage, steps 1-2: post-RoPE Q/K capture over calibration tokens and the SVD per KV head.

For each KV head h of layer l, the rows of every post-RoPE query head of its group and the rows of the
post-RoPE key head (one copy per query head of the group) are stacked into X = [Q_1; K; ...; Q_g; K].
X = U diag(sigma) R^T gives the singular values sigma and the rotation R (columns ordered by decreasing
singular value). X is never materialized: the Gram matrix X^T X is accumulated over the sampled
positions and decomposed, which yields the same R and sigma.

V is compressed as in Palu: W_V^h is decomposed by SVD after whitening with the statistics of its input
activations (Palu's default "whiten" decomposition, adapted from SVD-LLM, calibrated on WikiText-2).
The whitened decomposition has the same right singular vectors and singular values as the SVD of the V
activations V = E W_V^h, which is how it is computed here: the Gram matrix of V is accumulated over the
calibration text and decomposed. The same is done for the pre-RoPE K of the Palu baseline. The plain
SVD of the weights (Palu's "svd" option) is available with `whiten=False`.

Calibration sequences tile the context window: `seq_len`-token sequences are run with position offsets
0, seq_len, 2*seq_len, ... so that every position of the window appears in a forward pass, and
`num_positions` positions are sampled uniformly from all of them.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field

import torch
from safetensors.torch import load_file, save_file

from fastdc.models import DecoderRunner, ModelShape, model_shape, sdpa

DEFAULT_POSITIONS = 16384
DEFAULT_SEQ_LEN = 8192
DEFAULT_WHITEN_TOKENS = 65536
DEFAULT_WHITEN_SEQ_LEN = 2048


def wikitext_train() -> list[str]:
    """Paragraphs of the WikiText-2 training split (the calibration text of the whitened decomposition)."""
    from datasets import load_dataset

    data = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    return [t for t in data["text"] if t.strip()]


def canonical_signs(basis: torch.Tensor) -> torch.Tensor:
    """Flip each column so that its entry of largest magnitude is positive (a fixed sign convention)."""
    idx = basis.abs().argmax(dim=-2, keepdim=True)
    sign = torch.sign(torch.gather(basis, -2, idx))
    sign[sign == 0] = 1
    return basis * sign


def spectrum_from_gram(gram: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """gram [..., D, D] (float64) -> singular values [..., D] (descending) and basis [..., D, D] (columns)."""
    evals, evecs = torch.linalg.eigh(gram)
    evals, evecs = evals.flip(-1), evecs.flip(-1)
    sigma = evals.clamp_min(0).sqrt()
    return sigma, canonical_signs(evecs)


def weight_spectrum(weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """weights [..., hidden, D] -> singular values [..., D] and right singular vectors as columns [..., D, D]."""
    _, s, vh = torch.linalg.svd(weights.float(), full_matrices=False)
    return s, canonical_signs(vh.transpose(-1, -2))


@dataclass
class Calibration:
    """Singular values and rotations of one model (all tensors float32, on CPU)."""

    shape: ModelShape
    qk_sigma: torch.Tensor  # [L, Hkv, D]
    qk_basis: torch.Tensor  # [L, Hkv, D, D], R of each KV head
    v_sigma: torch.Tensor  # [L, Hkv, D], from W_V
    v_basis: torch.Tensor  # [L, Hkv, D, D]
    k_sigma: torch.Tensor  # [L, Hkv, D], from W_K (pre-RoPE baseline)
    k_basis: torch.Tensor  # [L, Hkv, D, D]
    meta: dict = field(default_factory=dict)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        tensors = {name: getattr(self, name).contiguous() for name in
                   ("qk_sigma", "qk_basis", "v_sigma", "v_basis", "k_sigma", "k_basis")}
        meta = {"shape": json.dumps(self.shape.to_dict()), "meta": json.dumps(self.meta)}
        save_file(tensors, path, metadata=meta)

    @classmethod
    def load(cls, path: str) -> "Calibration":
        from safetensors import safe_open

        with safe_open(path, "pt") as f:
            raw = f.metadata()
        tensors = load_file(path)
        return cls(shape=ModelShape(**json.loads(raw["shape"])), meta=json.loads(raw["meta"]), **tensors)


def random_tokens(tokenizer, num_tokens: int, generator: torch.Generator) -> torch.Tensor:
    """Tokens drawn uniformly from the regular vocabulary (special tokens excluded)."""
    vocab = torch.arange(tokenizer.vocab_size)
    special = torch.tensor(sorted(set(tokenizer.all_special_ids)), dtype=torch.long)
    if len(special):
        vocab = vocab[~torch.isin(vocab, special)]
    return vocab[torch.randint(len(vocab), (num_tokens,), generator=generator)]


def text_tokens(tokenizer, texts, num_tokens: int) -> torch.Tensor:
    """Concatenated tokens of `texts` in order, truncated to `num_tokens` (the texts repeat if they are shorter)."""
    texts = list(texts)
    if not texts:
        raise ValueError("no calibration texts")
    ids: list[int] = []
    while len(ids) < num_tokens:
        before = len(ids)
        for text in texts:
            ids.extend(tokenizer(text, add_special_tokens=False)["input_ids"])
            if len(ids) >= num_tokens:
                break
        if len(ids) == before:
            raise ValueError("the calibration texts contain no tokens")
    return torch.tensor(ids[:num_tokens], dtype=torch.long)


class _Whiten:
    """Attention function that accumulates the Gram matrices of V and of pre-RoPE K per KV head."""

    def __init__(self, shape: ModelShape, device: torch.device):
        size = (shape.num_layers, shape.num_kv_heads, shape.head_dim, shape.head_dim)
        self.v = torch.zeros(size, dtype=torch.float64, device=device)
        self.k = torch.zeros(size, dtype=torch.float64, device=device)
        self.shape = shape

    def __call__(self, layer: int, parts, hidden, cos, sin):
        q, k, v = parts.project(hidden)
        vv, kk = v[0].float(), k[0].float()
        self.v[layer] += torch.einsum("hnd,hne->hde", vv, vv).double()
        self.k[layer] += torch.einsum("hnd,hne->hde", kk, kk).double()
        q, k = parts.rope(q, k, cos, sin)
        out = sdpa(q, k, v, scale=self.shape.head_dim**-0.5, causal=True)
        return parts.o_proj(out.transpose(1, 2).reshape(hidden.shape[0], hidden.shape[1], -1))


class _Capture:
    """Attention function for DecoderRunner that accumulates the Gram matrices of X per KV head."""

    def __init__(self, shape: ModelShape, device: torch.device):
        self.shape = shape
        self.gram = torch.zeros(shape.num_layers, shape.num_kv_heads, shape.head_dim, shape.head_dim,
                                dtype=torch.float64, device=device)
        self.positions: torch.Tensor | None = None  # indices into the sequence dimension

    def __call__(self, layer: int, parts, hidden, cos, sin):
        s = self.shape
        q, k, v = parts.project(hidden)
        q, k = parts.rope(q, k, cos, sin)
        idx = self.positions
        qs = q[0, :, idx].float().view(s.num_kv_heads, s.group_size, -1, s.head_dim)
        ks = k[0, :, idx].float()
        gram = torch.einsum("hgnd,hgne->hde", qs, qs) + s.group_size * torch.einsum("hnd,hne->hde", ks, ks)
        self.gram[layer] += gram.double()
        out = sdpa(q, k, v, scale=s.head_dim**-0.5, causal=True)
        return parts.o_proj(out.transpose(1, 2).reshape(hidden.shape[0], hidden.shape[1], -1))


@torch.inference_mode()
def calibrate(model, tokenizer, num_positions: int = DEFAULT_POSITIONS, seq_len: int | None = None,
              texts=None, seed: int = 0, window: int | None = None, progress: bool = False,
              whiten: bool = True, whiten_texts=None, whiten_tokens: int = DEFAULT_WHITEN_TOKENS,
              whiten_seq_len: int = DEFAULT_WHITEN_SEQ_LEN) -> Calibration:
    """Run the offline stage (steps 1-2).

    Q/K: post-RoPE rows over random tokens (or over `texts` if given).
    V and pre-RoPE K: whitened decomposition over `whiten_texts` (default: WikiText-2 train), or the plain
    SVD of the weights if `whiten` is False.
    """
    shape = model_shape(model.config)
    window = window or shape.max_positions
    seq_len = min(seq_len or DEFAULT_SEQ_LEN, window)
    num_seqs = max(1, math.ceil(window / seq_len))
    gen = torch.Generator().manual_seed(seed)
    total = num_seqs * seq_len
    tokens = random_tokens(tokenizer, total, gen) if texts is None else text_tokens(tokenizer, texts, total)

    runner = DecoderRunner(model)
    capture = _Capture(shape, runner.device)
    # positions sampled uniformly over all num_seqs * seq_len tokens, then split per sequence
    picks = torch.randperm(total, generator=gen)[: min(num_positions, total)].sort().values
    for i in range(num_seqs):
        start = min(i * seq_len, max(0, shape.max_positions - seq_len))
        local = picks[(picks >= i * seq_len) & (picks < (i + 1) * seq_len)] - i * seq_len
        if len(local) == 0:
            continue
        ids = tokens[i * seq_len : (i + 1) * seq_len].unsqueeze(0).to(runner.device)
        pos = torch.arange(start, start + seq_len, device=runner.device).unsqueeze(0)
        last = int(local.max()) + 1  # later tokens cannot influence the sampled ones
        capture.positions = local.to(runner.device)
        runner.forward(ids[:, :last], pos[:, :last], capture, logits="none")
        if progress:
            print(f"calibration sequence {i + 1}/{num_seqs}: {len(local)} positions", flush=True)

    qk_sigma, qk_basis = spectrum_from_gram(capture.gram.cpu())
    meta = dict(source="random" if texts is None else "text", num_positions=int(len(picks)), seq_len=seq_len,
                num_sequences=num_seqs, window=window, seed=seed, model=getattr(model.config, "_name_or_path", ""),
                v_decomposition="whiten" if whiten else "svd")
    if whiten:
        texts_w = whiten_texts if whiten_texts is not None else wikitext_train()
        wseq = min(whiten_seq_len, shape.max_positions)
        count = max(1, whiten_tokens // wseq)
        ids = text_tokens(tokenizer, texts_w, count * wseq)
        acc = _Whiten(shape, runner.device)
        for i in range(count):
            chunk = ids[i * wseq : (i + 1) * wseq].unsqueeze(0).to(runner.device)
            runner.forward(chunk, torch.arange(wseq, device=runner.device).unsqueeze(0), acc, logits="none")
        v_sigma, v_basis = spectrum_from_gram(acc.v.cpu())
        k_sigma, k_basis = spectrum_from_gram(acc.k.cpu())
        meta.update(whiten_tokens=count * wseq, whiten_seq_len=wseq,
                    whiten_source="wikitext-2" if whiten_texts is None else "text")
    else:
        vs, vb, ks, kb = [], [], [], []
        for parts in runner.parts:
            wv = torch.stack([parts.kv_weight("v", h) for h in range(shape.num_kv_heads)])
            wk = torch.stack([parts.kv_weight("k", h) for h in range(shape.num_kv_heads)])
            for (sig, bas), out_s, out_b in ((weight_spectrum(wv), vs, vb), (weight_spectrum(wk), ks, kb)):
                out_s.append(sig.cpu())
                out_b.append(bas.cpu())
        v_sigma, v_basis, k_sigma, k_basis = torch.stack(vs), torch.stack(vb), torch.stack(ks), torch.stack(kb)
    return Calibration(shape=shape, qk_sigma=qk_sigma.float(), qk_basis=qk_basis.float(),
                       v_sigma=v_sigma.float(), v_basis=v_basis.float(),
                       k_sigma=k_sigma.float(), k_basis=k_basis.float(), meta=meta)
