"""Access to the attention layers of Hugging Face decoder-only models that use rotary position embeddings.

`DecoderRunner` executes such a model layer by layer with the original weights and calls a user-supplied
attention function in every layer. The offline stage uses it to capture post-RoPE Q and K, and the
reference engine uses it to run the compression methods.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

import torch
from torch import nn

SUPPORTED_MODEL_TYPES = ("llama", "mistral", "qwen2", "qwen3", "phi3")


@dataclass(frozen=True)
class ModelShape:
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    hidden_size: int
    max_positions: int

    @property
    def group_size(self) -> int:
        return self.num_heads // self.num_kv_heads

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def model_shape(config) -> ModelShape:
    heads = config.num_attention_heads
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // heads
    return ModelShape(
        num_layers=config.num_hidden_layers,
        num_heads=heads,
        num_kv_heads=getattr(config, "num_key_value_heads", None) or heads,
        head_dim=head_dim,
        hidden_size=config.hidden_size,
        max_positions=config.max_position_embeddings,
    )


def load_model(name_or_path: str, dtype: torch.dtype = torch.bfloat16, device: str | torch.device = "cuda"):
    """Load a supported model and its tokenizer with the original weights."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(name_or_path)
    model = AutoModelForCausalLM.from_pretrained(name_or_path, dtype=dtype, attn_implementation="sdpa")
    if model.config.model_type not in SUPPORTED_MODEL_TYPES:
        raise ValueError(f"model type {model.config.model_type!r} is not supported; supported: {SUPPORTED_MODEL_TYPES}")
    model.to(device).eval()
    return model, tokenizer


class AttentionParts:
    """Uniform view of one attention module: projections, optional Q/K norms and the rotary function."""

    def __init__(self, module: nn.Module, shape: ModelShape):
        self.module = module
        self.shape = shape
        self.fused = hasattr(module, "qkv_proj")
        self.q_norm = getattr(module, "q_norm", None)
        self.k_norm = getattr(module, "k_norm", None)
        self.o_proj = module.o_proj
        self._rope = sys.modules[type(module).__module__].apply_rotary_pos_emb

    def project(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """hidden [B, S, d] -> pre-RoPE q [B, H, S, D], k and v [B, Hkv, S, D] (Q/K norms applied)."""
        s = self.shape
        bsz, seq, _ = hidden.shape
        if self.fused:
            qkv = self.module.qkv_proj(hidden)
            nq, nk = s.num_heads * s.head_dim, s.num_kv_heads * s.head_dim
            q, k, v = qkv[..., :nq], qkv[..., nq : nq + nk], qkv[..., nq + nk :]
        else:
            q, k, v = self.module.q_proj(hidden), self.module.k_proj(hidden), self.module.v_proj(hidden)
        q = q.view(bsz, seq, s.num_heads, s.head_dim)
        k = k.view(bsz, seq, s.num_kv_heads, s.head_dim)
        v = v.view(bsz, seq, s.num_kv_heads, s.head_dim)
        if self.q_norm is not None:
            q = self.q_norm(q)
        if self.k_norm is not None:
            k = self.k_norm(k)
        return q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

    def rope(self, q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        return self._rope(q, k, cos, sin)

    def rope_keys(self, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        return self._rope(k, k, cos, sin)[1]

    def _weight_rows(self, which: str) -> torch.Tensor:
        s = self.shape
        if self.fused:
            w = self.module.qkv_proj.weight
            nq, nk = s.num_heads * s.head_dim, s.num_kv_heads * s.head_dim
            return {"q": w[:nq], "k": w[nq : nq + nk], "v": w[nq + nk :]}[which]
        return getattr(self.module, which + "_proj").weight

    def kv_weight(self, which: str, kv_head: int) -> torch.Tensor:
        """W_K^h or W_V^h as a [hidden, head_dim] matrix (k = E W)."""
        d = self.shape.head_dim
        return self._weight_rows(which)[kv_head * d : (kv_head + 1) * d].t()


class DecoderRunner:
    """Runs the decoder stack of a supported model with a custom attention function per layer.

    `attention(layer_idx, parts, hidden, cos, sin)` receives the normalized hidden states [B, S, d] and the
    rotary tables, and returns the output of the attention block [B, S, d] (after o_proj).
    """

    def __init__(self, model: nn.Module, mlp_chunk: int = 8192):
        self.model = model
        self.inner = model.model
        self.shape = model_shape(model.config)
        self.parts = [AttentionParts(layer.self_attn, self.shape) for layer in self.inner.layers]
        self.mlp_chunk = mlp_chunk

    @property
    def device(self) -> torch.device:
        return self.inner.embed_tokens.weight.device

    def _mlp(self, layer: nn.Module, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.shape[1] <= self.mlp_chunk:
            return layer.mlp(layer.post_attention_layernorm(hidden))
        out = torch.empty_like(hidden)
        for start in range(0, hidden.shape[1], self.mlp_chunk):
            part = hidden[:, start : start + self.mlp_chunk]
            out[:, start : start + self.mlp_chunk] = layer.mlp(layer.post_attention_layernorm(part))
        return out

    @torch.inference_mode()
    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor, attention, logits: str = "last",
                num_layers: int | None = None):
        """input_ids, positions: [B, S]. logits: "last" (logits of the last position), "all", or "none"."""
        hidden = self.inner.embed_tokens(input_ids)
        cos, sin = self.inner.rotary_emb(hidden, positions)
        layers = self.inner.layers if num_layers is None else self.inner.layers[:num_layers]
        for idx, layer in enumerate(layers):
            hidden = hidden + attention(idx, self.parts[idx], layer.input_layernorm(hidden), cos, sin)
            hidden = hidden + self._mlp(layer, hidden)
        if logits == "none":
            return None
        if logits == "last":
            hidden = hidden[:, -1:]
        return self.model.lm_head(self.inner.norm(hidden)).float()


def sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float, causal: bool,
         mask: torch.Tensor | None = None) -> torch.Tensor:
    """Scaled dot-product attention with grouped KV heads; q [B, H, S, Dq], k [B, Hkv, T, Dq], v [B, Hkv, T, Dv].

    Q/K and V widths may differ; both are zero-padded to a common multiple of 8 so that the fused SDPA
    kernels apply (zero columns do not change the result). `mask` [B, Hkv, S, T] (True = attend) replaces
    the causal mask when given.
    """
    width = max(q.shape[-1], v.shape[-1])
    width = (width + 7) // 8 * 8
    dv = v.shape[-1]

    def pad(x):
        return x if x.shape[-1] == width else nn.functional.pad(x, (0, width - x.shape[-1]))

    group = q.shape[1] // k.shape[1]
    q, k, v = pad(q), pad(k), pad(v)
    if group > 1:
        k = k.repeat_interleave(group, dim=1)
        v = v.repeat_interleave(group, dim=1)
    if mask is not None:
        if group > 1:
            mask = mask.repeat_interleave(group, dim=1)
        out = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale)
    else:
        out = nn.functional.scaled_dot_product_attention(q, k, v, is_causal=causal and q.shape[2] > 1, scale=scale)
    return out[..., :dv]
