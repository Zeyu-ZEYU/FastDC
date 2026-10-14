"""Reference engine: runs one request at a time on Hugging Face weights with a compression plan.

The engine is the accuracy path of this repository (calibration subsets, F_p tables, accuracy thresholds).
It stores the cache in the compressed form of each method and computes attention in PyTorch:

    full    post-RoPE K and V
    fastdc  post-RoPE K rotated by R and truncated; V projected onto the leading directions of the
            decomposition of W_V (V' = E A_V); queries are rotated by the same R; attention runs on the
            compressed Q, K and V and its output is mapped back through the V basis (the vLLM integration
            folds this mapping into o_proj)
    palu    pre-RoPE K projected onto the leading directions of the decomposition of W_K; before attention
            the cached K is reconstructed (decompressed) and RoPE is applied at the cached positions

Two optional modifiers act on the stored cache of any method:
    kvquant    4-bit quantization of the stored K (per channel, ranges calibrated offline) and V (per token)
    keyformer  token eviction with Gumbel-regularized accumulated attention scores and a recent window
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from fastdc.models import DecoderRunner, sdpa
from fastdc.plan import CompressionPlan


@dataclass(frozen=True)
class KVQuantConfig:
    bits: int = 4
    v_group: int = 32  # channels per V quantization group (per token)
    calib_tokens: int = 4096
    clip: float = 1.0  # K ranges are the calibrated min/max scaled by `clip`


@dataclass(frozen=True)
class KeyformerConfig:
    keep_ratio: float = 0.5  # retained tokens / processed tokens
    window_fraction: float = 0.25  # share of the budget reserved for the most recent tokens
    tau_start: float = 1.0
    tau_end: float = 2.0
    query_chunk: int = 256
    seed: int = 0


class _Layer:
    __slots__ = ("k", "v", "pos", "score")

    def __init__(self):
        self.k = self.v = self.pos = self.score = None


class Engine:
    def __init__(self, model, plan: CompressionPlan, kvquant: KVQuantConfig | None = None,
                 keyformer: KeyformerConfig | None = None, tokenizer=None):
        self.model = model
        self.plan = plan
        self.runner = DecoderRunner(model)
        self.shape = self.runner.shape
        self.device = self.runner.device
        self.scale = self.shape.head_dim ** -0.5
        self.kvquant = kvquant
        self.keyformer = keyformer
        s = self.shape
        self.k_basis, self.v_basis, self.k_mask, self.v_mask = [], [], [], []
        for layer in range(s.num_layers):
            if plan.method == "full":
                self.k_basis.append(None)
                self.v_basis.append(None)
            else:
                self.k_basis.append(plan.kept_k_basis(layer).float().to(self.device))
                self.v_basis.append(plan.kept_v_basis(layer).float().to(self.device))
            kw, vw = plan.k_width(layer), plan.v_width(layer)
            self.k_mask.append((torch.arange(kw) < plan.k_dims[layer].unsqueeze(1)).to(self.device))
            self.v_mask.append((torch.arange(vw) < plan.v_dims[layer].unsqueeze(1)).to(self.device))
        self.k_ranges = None
        self._gen = torch.Generator(device=self.device)
        self.reset()
        if kvquant is not None:
            if tokenizer is None:
                raise ValueError("kvquant needs the tokenizer for its offline range calibration")
            self._calibrate_kvquant(tokenizer)

    # ----------------------------------------------------------------- request state
    def reset(self) -> None:
        self.layers = [_Layer() for _ in range(self.shape.num_layers)]
        self.length = 0
        self.steps = 0
        self._collect_ranges = None
        self._unfed = None
        if self.keyformer is not None:
            self._gen.manual_seed(self.keyformer.seed)

    # ----------------------------------------------------------------- compression helpers
    def _to_store(self, layer: int, parts, q, k, v, cos, sin):
        """Returns (q for attention, K to store [Hkv, S, Wk], V to store [Hkv, S, Wv])."""
        method = self.plan.method
        if method == "palu":
            k_lat = torch.einsum("hsd,hdw->hsw", k[0].float(), self.k_basis[layer]).to(k.dtype)
            q, _ = parts.rope(q, k, cos, sin)
            k_store = k_lat
        else:
            q, k = parts.rope(q, k, cos, sin)
            if method == "fastdc":
                s = self.shape
                qg = q[0].float().view(s.num_kv_heads, s.group_size, q.shape[2], s.head_dim)
                q = torch.einsum("hgsd,hdw->hgsw", qg, self.k_basis[layer]).reshape(
                    1, s.num_heads, q.shape[2], -1).to(k.dtype)
                k_store = torch.einsum("hsd,hdw->hsw", k[0].float(), self.k_basis[layer]).to(k.dtype)
            else:
                k_store = k[0]
        v_store = v[0] if method == "full" else torch.einsum(
            "hsd,hdw->hsw", v[0].float(), self.v_basis[layer]).to(v.dtype)
        return q, k_store, v_store

    def _keys_for_attention(self, layer: int, parts, entry: _Layer) -> torch.Tensor:
        if self.plan.method != "palu":
            return entry.k.unsqueeze(0)
        k_pre = torch.einsum("hsw,hdw->hsd", entry.k.float(), self.k_basis[layer]).to(entry.k.dtype)
        cos, sin = self.runner.inner.rotary_emb(k_pre, entry.pos)  # positions [Hkv, T]: heads as batch
        return parts.rope_keys(k_pre.unsqueeze(1), cos, sin).squeeze(1).unsqueeze(0)

    def _output(self, layer: int, out: torch.Tensor) -> torch.Tensor:
        """Attention output [1, H, S, Wv] -> [1, H, S, D] in the original V coordinates."""
        if self.plan.method == "full":
            return out
        s = self.shape
        og = out[0].float().view(s.num_kv_heads, s.group_size, out.shape[2], -1)
        full = torch.einsum("hgsw,hdw->hgsd", og, self.v_basis[layer])
        return full.reshape(1, s.num_heads, out.shape[2], s.head_dim).to(out.dtype)

    # ----------------------------------------------------------------- kvquant
    def _quant_k(self, layer: int, k: torch.Tensor) -> torch.Tensor:
        lo, hi = self.k_ranges[layer]
        levels = 2 ** self.kvquant.bits - 1
        scale = ((hi - lo) / levels).clamp_min(1e-6)
        codes = ((k.float() - lo.unsqueeze(1)) / scale.unsqueeze(1)).round().clamp(0, levels)
        out = codes * scale.unsqueeze(1) + lo.unsqueeze(1)
        return (out * self.k_mask[layer].unsqueeze(1)).to(k.dtype)

    def _quant_v(self, layer: int, v: torch.Tensor) -> torch.Tensor:
        hkv, n, w = v.shape
        g = self.kvquant.v_group
        groups = math.ceil(w / g)
        pad = groups * g - w
        mask = self.v_mask[layer].unsqueeze(1).expand(hkv, n, w)
        x = torch.nn.functional.pad(v.float(), (0, pad)).view(hkv, n, groups, g)
        m = torch.nn.functional.pad(mask, (0, pad)).view(hkv, n, groups, g)
        lo = torch.where(m, x, torch.full_like(x, float("inf"))).amin(-1, keepdim=True)
        hi = torch.where(m, x, torch.full_like(x, float("-inf"))).amax(-1, keepdim=True)
        lo = torch.where(torch.isfinite(lo), lo, torch.zeros_like(lo))
        hi = torch.where(torch.isfinite(hi), hi, torch.zeros_like(hi))
        levels = 2 ** self.kvquant.bits - 1
        scale = ((hi - lo) / levels).clamp_min(1e-6)
        out = ((x - lo) / scale).round().clamp(0, levels) * scale + lo
        out = out.view(hkv, n, groups * g)[..., :w]
        return (out * self.v_mask[layer].unsqueeze(1)).to(v.dtype)

    def _calibrate_kvquant(self, tokenizer) -> None:
        from fastdc.calibration import random_tokens

        gen = torch.Generator().manual_seed(1234)
        ids = random_tokens(tokenizer, self.kvquant.calib_tokens, gen).unsqueeze(0).to(self.device)
        s = self.shape
        self.k_ranges = None
        self.reset()
        self._collect_ranges = [None] * s.num_layers
        pos = torch.arange(ids.shape[1], device=self.device).unsqueeze(0)
        self.runner.forward(ids, pos, self._attention, logits="none")
        ranges = []
        for layer, (lo, hi) in enumerate(self._collect_ranges):
            mid, half = (hi + lo) / 2, (hi - lo) / 2 * self.kvquant.clip
            mask = self.k_mask[layer].float()
            ranges.append(((mid - half) * mask, (mid + half) * mask))
        self.k_ranges = ranges
        self.reset()

    # ----------------------------------------------------------------- keyformer
    def _tau(self) -> float:
        kf = self.keyformer
        frac = min(1.0, self.steps / 512.0)
        return kf.tau_start + (kf.tau_end - kf.tau_start) * frac

    def _scores(self, q: torch.Tensor, k: torch.Tensor, causal: bool, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Accumulated Gumbel-softmax scores of every key [Hkv, T] over the given queries [1, H, S, W]."""
        s = self.shape
        tau = self._tau()
        T = k.shape[2]
        total = torch.zeros(s.num_kv_heads, T, device=q.device, dtype=torch.float32)
        kk = k[0].float()
        chunk = max(1, min(self.keyformer.query_chunk, int(5e7 // (q.shape[1] * T))))
        for start in range(0, q.shape[2], chunk):
            qc = q[0, :, start : start + chunk].float()
            qg = qc.view(s.num_kv_heads, s.group_size, qc.shape[1], -1)
            logits = torch.einsum("hgsw,htw->hgst", qg, kk) * self.scale
            if mask is not None:
                logits = logits.masked_fill(~mask[0, :, None, start : start + qc.shape[1]], float("-inf"))
            elif causal:
                rows = torch.arange(start, start + qc.shape[1], device=q.device) + (T - q.shape[2])
                logits = logits.masked_fill(torch.arange(T, device=q.device) > rows.unsqueeze(1), float("-inf"))
            u = torch.rand(logits.shape, device=q.device, generator=self._gen).clamp_(1e-10, 1 - 1e-10)
            gumbel = -torch.log(-torch.log(u))
            probs = torch.softmax((logits + gumbel) / tau, dim=-1)
            total += probs.sum(dim=(1, 2))
        return total

    def _evict(self, entry: _Layer) -> None:
        kf = self.keyformer
        T = entry.k.shape[1]
        budget = max(1, math.ceil(kf.keep_ratio * self.length))
        if T <= budget:
            return
        window = min(T, max(1, int(kf.window_fraction * budget)))
        keep_old = budget - window
        old_scores = entry.score[:, : T - window]
        top = old_scores.topk(keep_old, dim=1).indices.sort(dim=1).values if keep_old > 0 else \
            torch.empty(entry.k.shape[0], 0, dtype=torch.long, device=entry.k.device)
        recent = torch.arange(T - window, T, device=entry.k.device).unsqueeze(0).expand(entry.k.shape[0], -1)
        idx = torch.cat([top, recent], dim=1)
        entry.k = torch.gather(entry.k, 1, idx.unsqueeze(-1).expand(-1, -1, entry.k.shape[-1]))
        entry.v = torch.gather(entry.v, 1, idx.unsqueeze(-1).expand(-1, -1, entry.v.shape[-1]))
        entry.pos = torch.gather(entry.pos, 1, idx)
        entry.score = torch.gather(entry.score, 1, idx)

    # ----------------------------------------------------------------- attention of one layer
    def _attention(self, layer: int, parts, hidden, cos, sin):
        q, k, v = parts.project(hidden)
        n_new = hidden.shape[1]
        q, k_store, v_store = self._to_store(layer, parts, q, k, v, cos, sin)
        if self._collect_ranges is not None:
            kf = k_store.float()
            lo, hi = kf.amin(1), kf.amax(1)
            prev = self._collect_ranges[layer]
            self._collect_ranges[layer] = (lo, hi) if prev is None else (torch.minimum(prev[0], lo), torch.maximum(prev[1], hi))
        if self.k_ranges is not None:
            k_store, v_store = self._quant_k(layer, k_store), self._quant_v(layer, v_store)
        entry = self.layers[layer]
        positions = torch.arange(self.length, self.length + n_new, device=self.device)
        positions = positions.unsqueeze(0).expand(self.shape.num_kv_heads, -1)
        if entry.k is None:
            entry.k, entry.v, entry.pos = k_store, v_store, positions
        else:
            entry.k = torch.cat([entry.k, k_store], dim=1)
            entry.v = torch.cat([entry.v, v_store], dim=1)
            entry.pos = torch.cat([entry.pos, positions], dim=1)
        keys = self._keys_for_attention(layer, parts, entry)
        causal = n_new > 1
        mask = None
        if causal and entry.k.shape[1] != n_new:  # new tokens on top of a cached prefix
            mask = (entry.pos.unsqueeze(1) <= positions[:1].view(1, -1, 1)).unsqueeze(0)
        out = sdpa(q, keys, entry.v.unsqueeze(0), scale=self.scale, causal=causal, mask=mask)
        if self.keyformer is not None and self._collect_ranges is None:
            scores = self._scores(q, keys, causal, mask)
            if entry.score is None:
                entry.score = scores
            else:
                entry.score = torch.cat([entry.score, scores.new_zeros(scores.shape[0], n_new)], dim=1) + scores
        out = self._output(layer, out)
        return parts.o_proj(out.transpose(1, 2).reshape(hidden.shape[0], n_new, -1))

    # ----------------------------------------------------------------- public API
    @torch.inference_mode()
    def prefill(self, input_ids: torch.Tensor) -> torch.Tensor:
        """input_ids [S] or [1, S] -> logits of the last position [vocab]."""
        self.reset()
        ids = input_ids.view(1, -1).to(self.device)
        pos = torch.arange(ids.shape[1], device=self.device).unsqueeze(0)
        logits = self.runner.forward(ids, pos, self._attention, logits="last")
        self.length = ids.shape[1]
        self._after_step()
        return logits[0, -1]

    @torch.inference_mode()
    def extend(self, input_ids) -> torch.Tensor:
        """Append several tokens to the current request (e.g. the next turn of a conversation)."""
        ids = torch.as_tensor(input_ids).view(1, -1).to(self.device)
        pos = torch.arange(self.length, self.length + ids.shape[1], device=self.device).unsqueeze(0)
        logits = self.runner.forward(ids, pos, self._attention, logits="last")
        self.length += ids.shape[1]
        self.steps += 1
        self._after_step()
        return logits[0, -1]

    @torch.inference_mode()
    def decode(self, token: int) -> torch.Tensor:
        ids = torch.tensor([[token]], device=self.device)
        pos = torch.tensor([[self.length]], device=self.device)
        logits = self.runner.forward(ids, pos, self._attention, logits="last")
        self.length += 1
        self.steps += 1
        self._after_step()
        return logits[0, -1]

    def _after_step(self) -> None:
        if self.keyformer is None:
            return
        for entry in self.layers:
            self._evict(entry)

    @torch.inference_mode()
    def generate(self, input_ids, max_new_tokens: int, eos_ids=(), stop=None) -> list[int]:
        """Greedy decoding of a new request. `stop(tokens)` may end generation early."""
        return self._continue(self.prefill(torch.as_tensor(input_ids)), max_new_tokens, eos_ids, stop)

    @torch.inference_mode()
    def generate_turns(self, turns: list, max_new_tokens: int, eos_ids=()) -> list[list[int]]:
        """Multi-turn generation in one conversation: every turn's tokens extend the cache of the previous ones."""
        outputs: list[list[int]] = []
        pending: list[int] = []
        for i, ids in enumerate(turns):
            ids = [int(t) for t in torch.as_tensor(ids).view(-1).tolist()]
            logits = self.prefill(torch.tensor(ids)) if i == 0 else self.extend(torch.tensor(pending + ids))
            out = self._continue(logits, max_new_tokens, eos_ids, None)
            pending = [] if self._unfed is None else [self._unfed]
            outputs.append(out)
        return outputs

    def _continue(self, logits, max_new_tokens: int, eos_ids, stop) -> list[int]:
        """Greedy loop; the last generated token is left in `self._unfed` when it was not fed to the cache."""
        out: list[int] = []
        eos = set(eos_ids)
        self._unfed = None
        for _ in range(max_new_tokens):
            token = int(logits.argmax())
            if token in eos:
                break
            out.append(token)
            if (stop is not None and stop(out)) or len(out) == max_new_tokens:
                self._unfed = token
                break
            logits = self.decode(token)
        return out

    def cache_bytes(self) -> int:
        """Bytes of the stored cache of the current request in the method's storage format."""
        total = 0
        bits = self.kvquant.bits if self.kvquant is not None else 16
        for layer, entry in enumerate(self.layers):
            if entry.k is None:
                continue
            T = entry.k.shape[1]
            widths = int(self.plan.k_dims[layer].sum() + self.plan.v_dims[layer].sum())
            total += T * widths * bits // 8
            if self.kvquant is not None:  # per-token V scale and zero point per group (16 bits each)
                groups = sum(math.ceil(int(w) / self.kvquant.v_group) for w in self.plan.v_dims[layer])
                total += T * groups * 4
        return total

    def full_cache_bytes(self, num_tokens: int) -> int:
        s = self.shape
        return num_tokens * s.num_layers * s.num_kv_heads * s.head_dim * 2 * 2
