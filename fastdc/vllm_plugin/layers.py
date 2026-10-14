"""Per-layer view of a compression plan on one tensor-parallel rank, and the V-basis weight folding.

A rank holds a contiguous range of query heads and of KV heads (KV heads are replicated when there are
fewer KV heads than ranks), as in vLLM's `QKVParallelLinear`. Every rank stores the K and V widths of
its own KV heads; the page of a layer is sized for the rank with the widest heads so that all ranks
publish the same cache layout to the engine.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from fastdc.plan import CompressionPlan


@dataclass(frozen=True)
class HeadSlice:
    """Query heads [q_start, q_start + num_q) and KV heads [kv_start, kv_start + num_kv) of one rank."""

    q_start: int
    num_q: int
    kv_start: int
    num_kv: int

    @property
    def kv_heads(self) -> range:
        return range(self.kv_start, self.kv_start + self.num_kv)


def head_slice(num_heads: int, num_kv_heads: int, tp_size: int, tp_rank: int) -> HeadSlice:
    if num_heads % tp_size:
        raise ValueError(f"{num_heads} attention heads cannot be split over {tp_size} ranks")
    num_q = num_heads // tp_size
    if num_kv_heads >= tp_size:
        if num_kv_heads % tp_size:
            raise ValueError(f"{num_kv_heads} KV heads cannot be split over {tp_size} ranks")
        num_kv = num_kv_heads // tp_size
        return HeadSlice(tp_rank * num_q, num_q, tp_rank * num_kv, num_kv)
    if tp_size % num_kv_heads:
        raise ValueError(f"{num_kv_heads} KV heads cannot be replicated over {tp_size} ranks")
    return HeadSlice(tp_rank * num_q, num_q, tp_rank // (tp_size // num_kv_heads), 1)


def token_elems(plan: CompressionPlan, layer: int, tp_size: int) -> int:
    """Cache elements per token of `layer`: the largest sum of K and V widths over the ranks."""
    s = plan.shape
    widths = (plan.k_dims[layer] + plan.v_dims[layer]).tolist()
    slices = (head_slice(s.num_heads, s.num_kv_heads, tp_size, r) for r in range(tp_size))
    return max(sum(widths[h] for h in heads.kv_heads) for heads in slices)


@dataclass
class LayerPlan:
    """What one attention layer of this rank stores, and the bases it projects with."""

    index: int
    method: str
    heads: HeadSlice
    head_dim: int
    group_size: int
    k_dims: tuple[int, ...]
    v_dims: tuple[int, ...]
    token_elems: int
    k_basis: torch.Tensor | None  # [num_kv, D, Wk] float32: R (fastdc) or the W_K basis (palu); zero beyond a width
    v_basis: torch.Tensor | None  # [num_kv, D, Wv] float32, zero beyond a head's width

    @property
    def k_width(self) -> int:
        return max(self.k_dims)

    @property
    def v_width(self) -> int:
        return max(self.v_dims)


def layer_plan(plan: CompressionPlan, layer: int, heads: HeadSlice, tp_size: int) -> LayerPlan:
    kv = list(heads.kv_heads)
    k_dims = tuple(int(plan.k_dims[layer, h]) for h in kv)
    v_dims = tuple(int(plan.v_dims[layer, h]) for h in kv)
    k_basis = v_basis = None
    if plan.method != "full":
        k_basis = plan.k_basis[layer, kv][..., : max(k_dims)].float().clone()
        v_basis = plan.v_basis[layer, kv][..., : max(v_dims)].float().clone()
        for i, (kd, vd) in enumerate(zip(k_dims, v_dims)):
            k_basis[i, :, kd:] = 0
            v_basis[i, :, vd:] = 0
    return LayerPlan(index=layer, method=plan.method, heads=heads, head_dim=plan.shape.head_dim,
                     group_size=plan.shape.group_size, k_dims=k_dims, v_dims=v_dims,
                     token_elems=token_elems(plan, layer, tp_size), k_basis=k_basis, v_basis=v_basis)


def check_projections(parent: nn.Module, lp: LayerPlan) -> None:
    """The attention block must have unquantized 16-bit fused QKV and output projections of the planned shape."""
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod

    qkv, o = getattr(parent, "qkv_proj", None), getattr(parent, "o_proj", None)
    if qkv is None or o is None:
        raise ValueError(f"{type(parent).__name__} has no fused qkv_proj/o_proj; this model is not supported")
    for proj in (qkv, o):
        if not isinstance(getattr(proj, "quant_method", None), UnquantizedLinearMethod):
            raise ValueError("quantized models are not supported by the FastDC backend")
        if proj.weight.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError(f"weights of dtype {proj.weight.dtype} are not supported; use float16 or bfloat16")
    d, h = lp.head_dim, lp.heads
    if qkv.weight.shape[0] != (h.num_q + 2 * h.num_kv) * d or o.weight.shape[1] != h.num_q * d:
        raise ValueError(f"projections of {type(parent).__name__} do not match the plan's head layout")


@torch.no_grad()
def fold_v_basis(parent: nn.Module, lp: LayerPlan) -> None:
    """Fold the kept V basis of every KV head into the V rows of qkv_proj and the input columns of o_proj.

    For KV head h with kept basis B (D x w, zero beyond the head's width), the V rows W_h (D x hidden) of
    qkv_proj become [B^T W_h; 0], so qkv_proj emits V' = V B in the first w features of the head. For query
    head i of group h, the input columns C_i (hidden x D) of o_proj become [C_i B, 0], so o_proj consumes
    the attention output in these coordinates directly.
    """
    d, h = lp.head_dim, lp.heads
    qkv, o = parent.qkv_proj, parent.o_proj
    basis = lp.v_basis.to(qkv.weight.device)  # [num_kv, D, Wv]
    wv = basis.shape[-1]

    v0 = (h.num_q + h.num_kv) * d
    rows = qkv.weight[v0 : v0 + h.num_kv * d]
    folded = torch.zeros(h.num_kv, d, rows.shape[1], device=rows.device)
    folded[:, :wv] = torch.einsum("hdw,hdc->hwc", basis, rows.float().view(h.num_kv, d, -1))
    rows.copy_(folded.view_as(rows))
    if getattr(qkv, "bias", None) is not None:
        bias = qkv.bias[v0 : v0 + h.num_kv * d]
        folded_bias = torch.zeros(h.num_kv, d, device=bias.device)
        folded_bias[:, :wv] = torch.einsum("hdw,hd->hw", basis, bias.float().view(h.num_kv, d))
        bias.copy_(folded_bias.view_as(bias))

    group = (h.q_start + torch.arange(h.num_q)) // lp.group_size - h.kv_start  # local KV head of each query head
    cols = o.weight.float().view(o.weight.shape[0], h.num_q, d)
    folded_o = torch.zeros_like(cols)
    folded_o[..., :wv] = torch.einsum("cnd,ndw->cnw", cols, basis[group.to(basis.device)])
    o.weight.copy_(folded_o.view_as(o.weight))
