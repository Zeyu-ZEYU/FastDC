import math

import pytest
import torch

from conftest import tiny_model
from fastdc.calibration import Calibration, calibrate, canonical_signs, spectrum_from_gram
from fastdc.engine import Engine, KeyformerConfig, KVQuantConfig
from fastdc.models import model_shape
from fastdc.plan import CompressionPlan, adaptive, full_plan, uniform
from fastdc.rates import kept_dims, kv_compression_rate, removed_energy


# ----------------------------------------------------------------------------- rates
def test_kept_dims_energy_rule():
    sigma = torch.tensor([[4.0, 2.0, 1.0, 1.0]])  # energies 16, 4, 1, 1 (total 22)
    assert kept_dims(sigma, 0.0).item() == 4
    assert kept_dims(sigma, 1 / 22).item() == 3
    assert kept_dims(sigma, 2 / 22).item() == 2
    assert kept_dims(sigma, 6 / 22).item() == 1
    assert kept_dims(sigma, 1.0).item() == 1  # at least one dimension is kept
    assert removed_energy(sigma, torch.tensor([2])).item() == pytest.approx(2 / 22)


def test_kept_dims_monotonic():
    sigma = torch.rand(3, 4, 32).sort(dim=-1, descending=True).values
    previous = None
    for r in (0.0, 0.01, 0.05, 0.2, 0.5):
        k = kept_dims(sigma, r)
        if previous is not None:
            assert bool((k <= previous).all())
        previous = k


def test_kv_rate():
    k = torch.full((2, 2), 64)
    assert kv_compression_rate(k, k, 128) == pytest.approx(0.5)


# ----------------------------------------------------------------------------- spectra
def test_spectrum_from_gram_matches_svd():
    torch.manual_seed(0)
    x = torch.randn(500, 16, dtype=torch.float64) @ torch.diag(torch.linspace(3, 0.1, 16, dtype=torch.float64))
    sigma, basis = spectrum_from_gram(x.T @ x)
    _, s, vh = torch.linalg.svd(x, full_matrices=False)
    assert torch.allclose(sigma, s, rtol=1e-6)
    ref = canonical_signs(vh.T)
    assert torch.allclose(basis, ref, atol=1e-6)


def test_calibration_shapes_and_orthogonality(model_type, tokenizer):
    model = tiny_model(model_type)
    cal = calibrate(model, tokenizer, num_positions=256, seq_len=128, window=256, whiten=False)
    s = cal.shape
    assert cal.qk_basis.shape == (s.num_layers, s.num_kv_heads, s.head_dim, s.head_dim)
    eye = torch.eye(s.head_dim)
    for basis in (cal.qk_basis, cal.v_basis, cal.k_basis):
        assert torch.allclose(basis.transpose(-1, -2) @ basis, eye.expand_as(basis), atol=1e-4)
    assert bool((cal.qk_sigma[..., :-1] >= cal.qk_sigma[..., 1:] - 1e-4).all())


def test_whitened_calibration(tokenizer):
    model = tiny_model("llama")
    cal = calibrate(model, tokenizer, num_positions=128, seq_len=128, window=128,
                    whiten_texts=["some calibration text " * 20], whiten_tokens=512, whiten_seq_len=128)
    assert cal.meta["v_decomposition"] == "whiten"
    assert torch.isfinite(cal.v_sigma).all()


def test_calibration_roundtrip(tmp_path, tokenizer):
    cal = calibrate(tiny_model("llama"), tokenizer, num_positions=64, seq_len=64, window=128, whiten=False)
    path = str(tmp_path / "c.safetensors")
    cal.save(path)
    loaded = Calibration.load(path)
    assert loaded.shape == cal.shape and torch.equal(loaded.qk_basis, cal.qk_basis)


# ----------------------------------------------------------------------------- plans
def test_plan_masks_and_roundtrip(tmp_path, tokenizer):
    cal = calibrate(tiny_model("llama"), tokenizer, num_positions=64, seq_len=64, window=128, whiten=False)
    plan = adaptive(cal, 0.05)
    for layer in range(plan.shape.num_layers):
        basis = plan.kept_k_basis(layer)
        for h in range(plan.shape.num_kv_heads):
            k = int(plan.k_dims[layer, h])
            assert torch.count_nonzero(basis[h, :, k:]) == 0
    path = str(tmp_path / "p.safetensors")
    plan.save(path)
    again = CompressionPlan.load(path)
    assert torch.equal(again.k_dims, plan.k_dims) and again.info["r"] == 0.05
    u = uniform(cal, 0.5, "palu")
    assert u.kv_rate == pytest.approx(0.5)


# ----------------------------------------------------------------------------- engine
def _ids(n=24, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(2, 96, (n,), generator=g)


@torch.inference_mode()
def test_full_engine_matches_transformers(model_type):
    model = tiny_model(model_type)
    ids = _ids()
    ref = model(ids.unsqueeze(0)).logits[0, -1]
    engine = Engine(model, full_plan(model_shape(model.config)))
    assert torch.allclose(engine.prefill(ids), ref, atol=1e-4)
    # greedy decoding equals transformers' generate
    out = engine.generate(ids, 8)
    hf = model.generate(ids.unsqueeze(0), max_new_tokens=8, do_sample=False, min_new_tokens=8)[0, len(ids):]
    assert out == hf.tolist()


@torch.inference_mode()
@pytest.mark.parametrize("method", ["fastdc", "palu"])
def test_no_removal_equals_full(model_type, tokenizer, method):
    model = tiny_model(model_type)
    cal = calibrate(model, tokenizer, num_positions=128, seq_len=64, window=128, whiten=False)
    ids = _ids()
    full = Engine(model, full_plan(cal.shape))
    plan = adaptive(cal, 0.0, method)
    assert plan.kv_rate == 0.0
    comp = Engine(model, plan)
    assert torch.allclose(comp.prefill(ids), full.prefill(ids), atol=1e-3)
    assert comp.generate(ids, 6) == full.generate(ids, 6)


@torch.inference_mode()
def test_fastdc_attention_equals_projection(tokenizer):
    """FastDC scores equal q R_k R_k^T k^T: compare against an explicit projection of Q and K."""
    model = tiny_model("llama")
    cal = calibrate(model, tokenizer, num_positions=128, seq_len=64, window=128, whiten=False)
    plan = adaptive(cal, 0.2)
    engine = Engine(model, plan)
    parts = engine.runner.parts[0]
    x = torch.randn(1, 10, model.config.hidden_size)
    pos = torch.arange(10).unsqueeze(0)
    cos, sin = engine.runner.inner.rotary_emb(x, pos)
    q, k, v = parts.project(x)
    q2, k_store, _ = engine._to_store(0, parts, q, k, v, cos, sin)
    qr, kr = parts.rope(q, k, cos, sin)
    basis = plan.kept_k_basis(0)
    s = model_shape(model.config)
    proj = basis @ basis.transpose(-1, -2)  # [Hkv, D, D]
    ref_scores = torch.einsum("hgsd,hde,hte->hgst", qr[0].view(s.num_kv_heads, s.group_size, 10, -1), proj, kr[0])
    got = torch.einsum("hgsw,htw->hgst", q2[0].view(s.num_kv_heads, s.group_size, 10, -1), k_store)
    assert torch.allclose(got, ref_scores, atol=1e-4)


@torch.inference_mode()
def test_extend_equals_single_prefill(model_type):
    model = tiny_model(model_type)
    ids = _ids(30)
    a = Engine(model, full_plan(model_shape(model.config)))
    whole = a.prefill(ids)
    a.prefill(ids[:20])
    part = a.extend(ids[20:])
    assert torch.allclose(whole, part, atol=1e-4)


@torch.inference_mode()
def test_multi_turn_generation(tokenizer):
    model = tiny_model("qwen3")
    engine = Engine(model, full_plan(model_shape(model.config)))
    outs = engine.generate_turns([_ids(12, 1), _ids(5, 2), _ids(5, 3)], 4)
    assert [len(o) for o in outs] == [4, 4, 4]
    assert engine.length == 12 + 4 + 5 + 4 + 5 + 3  # the last token of the last turn is not fed


@torch.inference_mode()
def test_keyformer_budget_and_kvquant_rate(tokenizer):
    model = tiny_model("llama")
    cal = calibrate(model, tokenizer, num_positions=128, seq_len=64, window=128, whiten=False)
    plan = adaptive(cal, 0.05)
    kf = Engine(model, plan, keyformer=KeyformerConfig(keep_ratio=0.5))
    kf.generate(_ids(40), 10)
    assert kf.layers[0].k.shape[1] == math.ceil(0.5 * kf.length)
    kq = Engine(model, plan, kvquant=KVQuantConfig(), tokenizer=tokenizer)
    kq.generate(_ids(40), 4)
    stored = kq.cache_bytes()
    assert stored < kq.full_cache_bytes(kq.length) * (1 - plan.kv_rate) * 0.5
