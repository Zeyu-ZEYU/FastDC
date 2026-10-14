"""Compression plans: which dimensions every KV head keeps, for one method and one operating point.

Methods
    full    no compression
    fastdc  post-RoPE Q and K rotated by the per-KV-head R from the offline stage and truncated to the head's
            QK width; V projected onto the leading directions of the decomposition of W_V (V' = E A_V).
            Attention runs on the compressed Q, K and V; the V basis is folded into the output projection.
    palu    pre-RoPE K projected onto the leading directions of the decomposition of W_K and V as above; the
            cache holds the low-rank K, which is reconstructed (decompressed) and rotated by RoPE before
            attention.

Adaptive plans derive every head's widths from one energy removal rate r (`adaptive`); uniform plans
give every head the same width (`uniform`).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import torch
from safetensors.torch import load_file, save_file

from fastdc.calibration import Calibration
from fastdc.models import ModelShape
from fastdc.rates import kept_dims, kv_compression_rate, uniform_dims

METHODS = ("full", "fastdc", "palu")


@dataclass
class CompressionPlan:
    method: str
    shape: ModelShape
    k_dims: torch.Tensor  # [L, Hkv] kept QK (fastdc) or K (palu) dimensions
    v_dims: torch.Tensor  # [L, Hkv]
    k_basis: torch.Tensor | None  # [L, Hkv, D, D] columns = directions (R for fastdc, W_K basis for palu)
    v_basis: torch.Tensor | None  # [L, Hkv, D, D]
    info: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.method not in METHODS:
            raise ValueError(f"unknown method {self.method!r}; choose from {METHODS}")

    @property
    def kv_rate(self) -> float:
        return kv_compression_rate(self.k_dims, self.v_dims, self.shape.head_dim)

    def k_width(self, layer: int) -> int:
        return int(self.k_dims[layer].max())

    def v_width(self, layer: int) -> int:
        return int(self.v_dims[layer].max())

    def kept_k_basis(self, layer: int) -> torch.Tensor:
        """[Hkv, D, Wk] with columns beyond each head's width set to zero (Wk = widest head of the layer)."""
        return _masked(self.k_basis[layer], self.k_dims[layer], self.k_width(layer))

    def kept_v_basis(self, layer: int) -> torch.Tensor:
        return _masked(self.v_basis[layer], self.v_dims[layer], self.v_width(layer))

    def summary(self) -> dict:
        return dict(method=self.method, kv_rate=round(self.kv_rate, 4),
                    k_dims_mean=float(self.k_dims.float().mean()), v_dims_mean=float(self.v_dims.float().mean()),
                    k_dims_min=int(self.k_dims.min()), k_dims_max=int(self.k_dims.max()),
                    v_dims_min=int(self.v_dims.min()), v_dims_max=int(self.v_dims.max()), **self.info)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        tensors = {"k_dims": self.k_dims.contiguous(), "v_dims": self.v_dims.contiguous()}
        if self.k_basis is not None:
            tensors["k_basis"] = self.k_basis.contiguous()
            tensors["v_basis"] = self.v_basis.contiguous()
        meta = {"method": self.method, "shape": json.dumps(self.shape.to_dict()), "info": json.dumps(self.info)}
        save_file(tensors, path, metadata=meta)

    @classmethod
    def load(cls, path: str) -> "CompressionPlan":
        from safetensors import safe_open

        with safe_open(path, "pt") as f:
            raw = f.metadata()
        t = load_file(path)
        return cls(method=raw["method"], shape=ModelShape(**json.loads(raw["shape"])), k_dims=t["k_dims"],
                   v_dims=t["v_dims"], k_basis=t.get("k_basis"), v_basis=t.get("v_basis"),
                   info=json.loads(raw["info"]))


def _masked(basis: torch.Tensor, dims: torch.Tensor, width: int) -> torch.Tensor:
    cols = torch.arange(width)
    mask = (cols.unsqueeze(0) < dims.unsqueeze(1)).to(basis.dtype)  # [Hkv, W]
    return basis[..., :width] * mask.unsqueeze(1)


def full_plan(shape: ModelShape) -> CompressionPlan:
    dims = torch.full((shape.num_layers, shape.num_kv_heads), shape.head_dim, dtype=torch.long)
    return CompressionPlan("full", shape, dims, dims.clone(), None, None)


def adaptive(calib: Calibration, r: float, method: str = "fastdc", r_v: float | None = None) -> CompressionPlan:
    """Per-head widths from one energy removal rate r (shared by QK/K and V unless r_v is given)."""
    k_sigma, k_basis = (calib.qk_sigma, calib.qk_basis) if method == "fastdc" else (calib.k_sigma, calib.k_basis)
    k_dims = kept_dims(k_sigma, r)
    v_dims = kept_dims(calib.v_sigma, r if r_v is None else r_v)
    return CompressionPlan(method, calib.shape, k_dims, v_dims, k_basis, calib.v_basis,
                           info=dict(mode="adaptive", r=r, r_v=r if r_v is None else r_v))


def uniform(calib: Calibration, p: float, method: str = "fastdc") -> CompressionPlan:
    """Every head keeps round((1 - p) * D) dimensions of K (or QK) and V."""
    s = calib.shape
    k_basis = calib.qk_basis if method == "fastdc" else calib.k_basis
    dims = uniform_dims((s.num_layers, s.num_kv_heads), s.head_dim, p)
    return CompressionPlan(method, s, dims, dims.clone(), k_basis, calib.v_basis, info=dict(mode="uniform", p=p))
