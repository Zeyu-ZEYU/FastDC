"""Rotary position embedding of a vLLM attention layer, applied and inverted in float32.

The cos/sin table is read from the layer's `RotaryEmbedding` module (`cos_sin_cache`, rows
[cos | sin] per position), so the rotation matches the one the model applies. The inverse divides by
cos^2 + sin^2, which also undoes tables that carry a magnitude scale (e.g. YaRN).
"""

from __future__ import annotations

import torch
from torch import nn

_FORWARD_METHODS = ("forward_static", "forward_native", "forward_cuda")


def _is_table_rotary(module: nn.Module) -> bool:
    """True for vLLM rotary embeddings whose forward is the plain table lookup by token position."""
    from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding

    cls = type(module)
    return isinstance(module, RotaryEmbedding) and all(
        getattr(cls, name) is getattr(RotaryEmbedding, name) for name in _FORWARD_METHODS
    )


class Rotary:
    def __init__(self, module: nn.Module, head_dim: int):
        if not _is_table_rotary(module):
            raise ValueError(f"rotary embedding {type(module).__name__} is not supported by the palu method")
        if module.rotary_dim > head_dim or module.rotary_dim % 2:
            raise ValueError(f"rotary dimension {module.rotary_dim} does not fit head dimension {head_dim}")
        self.module = module
        self.rotary_dim = int(module.rotary_dim)
        self.neox = bool(module.is_neox_style)

    def cos_sin(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """positions [...] -> cos, sin [..., rotary_dim / 2] in float32 (positions are clamped to the table)."""
        table = self.module.cos_sin_cache
        pos = positions.reshape(-1).clamp(0, table.shape[0] - 1)
        cos, sin = table.index_select(0, pos).float().chunk(2, dim=-1)
        shape = (*positions.shape, self.rotary_dim // 2)
        return cos.reshape(shape), sin.reshape(shape)

    def rotate(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, inverse: bool = False) -> torch.Tensor:
        """Rotate the leading `rotary_dim` features of x [..., D] (float32); cos/sin broadcast to x[..., :rot/2]."""
        rot = self.rotary_dim
        if self.neox:
            x1, x2 = x[..., : rot // 2], x[..., rot // 2 : rot]
        else:
            x1, x2 = x[..., 0:rot:2], x[..., 1:rot:2]
        if inverse:
            norm = cos * cos + sin * sin
            o1 = (x1 * cos + x2 * sin) / norm
            o2 = (x2 * cos - x1 * sin) / norm
        else:
            o1 = x1 * cos - x2 * sin
            o2 = x2 * cos + x1 * sin
        rotated = torch.cat((o1, o2), dim=-1) if self.neox else torch.stack((o1, o2), dim=-1).flatten(-2)
        if rot == x.shape[-1]:
            return rotated
        return torch.cat((rotated, x[..., rot:]), dim=-1)
