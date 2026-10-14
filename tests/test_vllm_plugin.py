"""Tests of the vLLM integration (fastdc.vllm_plugin).

Layer level: the weight folding, the tensor-parallel head slices, the rotary embedding helper, the palu
kernels and the attention of one layer driven step by step with vLLM's metadata (chunked prefill, mixed
decode and prefill batches, padded token rows) are compared with direct computations.

Engine level (one GPU; FASTDC_TEST_MODEL, default Qwen/Qwen3-0.6B), every engine in its own process:
  (a) the full plan on the FastDC backend follows vLLM's default backend,
  (b) a fastdc plan that keeps every dimension follows the full plan,
  (c) compressed fastdc and palu plans follow the Hugging Face reference engine (fastdc.engine),
  (d) the KV cache capacity reported by vLLM grows with the compression,
  (e) chunked prefill and prefix caching give the same tokens as one-pass prefill without caching,
  (f) with torch.compile and piecewise CUDA graphs (vLLM's default mode) the tokens are those of eager mode.
Two greedy generations agree when every token of one is the other's choice or within a near tie of it under
the reference log-probabilities. The calibration of the test model is cached in FASTDC_TEST_CACHE
(default ~/.cache/fastdc/tests).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
pytest.importorskip("vllm")

from fastdc.kernels.layout import layer_layout  # noqa: E402
from fastdc.models import ModelShape  # noqa: E402
from fastdc.plan import CompressionPlan  # noqa: E402
from fastdc.vllm_plugin.layers import fold_v_basis, head_slice, layer_plan, token_elems  # noqa: E402
from fastdc.vllm_plugin.settings import PluginSettings, settings_from  # noqa: E402

DEVICE = torch.device("cuda")


# ------------------------------------------------------------------------------------------------ helpers
def _orthonormal(*batch: int, dim: int, generator: torch.Generator) -> torch.Tensor:
    q, _ = torch.linalg.qr(torch.randn(*batch, dim, dim, generator=generator, dtype=torch.float64))
    return q.float()


def _plan(method: str, k_dims, v_dims, num_heads: int = 4, head_dim: int = 64, seed: int = 0) -> CompressionPlan:
    """A one-layer plan with random orthonormal bases; k_dims / v_dims hold one width per KV head."""
    gen = torch.Generator().manual_seed(seed)
    hkv = len(k_dims)
    shape = ModelShape(num_layers=1, num_heads=num_heads, num_kv_heads=hkv, head_dim=head_dim,
                       hidden_size=num_heads * head_dim, max_positions=4096)
    k, v = torch.tensor([k_dims]), torch.tensor([v_dims])
    if method == "full":
        return CompressionPlan("full", shape, k, v, None, None)
    return CompressionPlan(method, shape, k, v, _orthonormal(1, hkv, dim=head_dim, generator=gen),
                           _orthonormal(1, hkv, dim=head_dim, generator=gen))


# ------------------------------------------------------------------------------------------------ settings
def test_settings_from_additional_config_and_environment(monkeypatch):
    monkeypatch.setenv("FASTDC_PLAN", "env.safetensors")
    monkeypatch.setenv("FASTDC_BALANCE", "0")
    s = settings_from({"attention": "reference"})
    assert s == PluginSettings(plan="env.safetensors", attention="reference", balance=False)
    s = settings_from({"plan": "cfg.safetensors", "balance": "true", "decompress_tokens": 4096})
    assert (s.plan, s.balance, s.decompress_tokens, s.attention) == ("cfg.safetensors", True, 4096, "kernels")
    monkeypatch.delenv("FASTDC_PLAN")
    with pytest.raises(ValueError, match="compression plan"):
        settings_from({})
    with pytest.raises(ValueError, match="unknown FastDC options"):
        settings_from({"plan": "p", "rate": 0.1})
    with pytest.raises(ValueError, match="attention"):
        settings_from({"plan": "p", "attention": "flash"})


# ------------------------------------------------------------------------------------------------ tensor parallelism
def test_step_profiler_writes_one_line_per_step(tmp_path):
    from fastdc.vllm_plugin.runtime import StepProfiler

    path = tmp_path / "profile.jsonl"
    prof = StepProfiler(str(path))
    x = torch.randn(256, 256, device=DEVICE)
    for step in range(3):
        prof.begin_step(num_reqs=2, num_tokens=10 + step)
        with prof.region("compress"):
            x = x @ x.T / 256
        with prof.region("schedule"):
            x = x + 1
    prof.flush()
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert [line["num_tokens"] for line in lines] == [10, 11, 12]
    assert all(line["compress_s"] > 0 and line["schedule_s"] > 0 and line["decompress_s"] == 0 for line in lines)
    disabled = StepProfiler("")
    with disabled.region("compress"):
        pass
    disabled.begin_step(1, 1)
    disabled.flush()


def test_head_slices_follow_vllm_sharding():
    assert head_slice(16, 8, 1, 0) == head_slice(16, 8, 1, 0)
    slices = [head_slice(16, 8, 4, r) for r in range(4)]
    assert [(s.q_start, s.num_q, s.kv_start, s.num_kv) for s in slices] == [(0, 4, 0, 2), (4, 4, 2, 2),
                                                                         (8, 4, 4, 2), (12, 4, 6, 2)]
    replicated = [head_slice(32, 4, 8, r) for r in range(8)]  # two ranks share every KV head
    assert [s.kv_start for s in replicated] == [0, 0, 1, 1, 2, 2, 3, 3]
    assert all((s.q_start // 8) == s.kv_start for s in replicated)  # query heads stay in their KV group
    with pytest.raises(ValueError):
        head_slice(16, 8, 3, 0)


def test_pages_are_sized_for_the_widest_rank():
    plan = _plan("fastdc", [10, 20, 30, 40], [5, 5, 5, 5], num_heads=8)
    assert token_elems(plan, 0, 1) == 10 + 20 + 30 + 40 + 20
    assert token_elems(plan, 0, 2) == max(10 + 20 + 10, 30 + 40 + 10)
    assert token_elems(plan, 0, 8) == 40 + 5  # one replicated KV head per rank


def _attention_block(hidden: int, heads: int, kv_heads: int, head_dim: int, bias: bool, gen: torch.Generator):
    block = torch.nn.Module()
    block.qkv_proj = torch.nn.Linear(hidden, (heads + 2 * kv_heads) * head_dim, bias=bias)
    block.o_proj = torch.nn.Linear(heads * head_dim, hidden, bias=False)
    with torch.no_grad():
        for p in block.parameters():
            p.copy_(torch.randn(p.shape, generator=gen) * 0.1)
    return block


@pytest.mark.parametrize("tp_size", [1, 2])
def test_v_basis_folding(tp_size):
    """qkv_proj emits V B in the first w features of a head; o_proj consumes [a, 0] as the original does a B^T."""
    gen = torch.Generator().manual_seed(0)
    heads, kv_heads, d, hidden = 8, 4, 16, 32
    plan = _plan("fastdc", [16, 16, 16, 16], [5, 9, 16, 12], num_heads=heads, head_dim=d)
    full = _attention_block(hidden, heads, kv_heads, d, True, gen)
    x = torch.randn(7, hidden, generator=gen)
    attn = torch.randn(7, heads, plan.v_width(0), generator=gen)
    v_basis = plan.kept_v_basis(0)  # [Hkv, D, W], zero beyond each head's width
    group = torch.arange(heads) // (heads // kv_heads)
    for rank in range(tp_size):
        hs = head_slice(heads, kv_heads, tp_size, rank)
        q_rows = slice(hs.q_start * d, (hs.q_start + hs.num_q) * d)
        k_rows = slice((heads + hs.kv_start) * d, (heads + hs.kv_start + hs.num_kv) * d)
        v_rows = slice((heads + kv_heads + hs.kv_start) * d, (heads + kv_heads + hs.kv_start + hs.num_kv) * d)
        block = torch.nn.Module()
        block.qkv_proj = torch.nn.Linear(hidden, (hs.num_q + 2 * hs.num_kv) * d)
        block.o_proj = torch.nn.Linear(hs.num_q * d, hidden, bias=False)
        with torch.no_grad():  # the rank's shards, as vLLM's QKVParallelLinear / RowParallelLinear hold them
            block.qkv_proj.weight.copy_(torch.cat([full.qkv_proj.weight[r] for r in (q_rows, k_rows, v_rows)]))
            block.qkv_proj.bias.copy_(torch.cat([full.qkv_proj.bias[r] for r in (q_rows, k_rows, v_rows)]))
            block.o_proj.weight.copy_(full.o_proj.weight[:, q_rows])
        lp = layer_plan(plan, 0, hs, tp_size)
        fold_v_basis(block, lp)
        with torch.no_grad():
            v_out = block.qkv_proj(x)[:, (hs.num_q + hs.num_kv) * d :].view(7, hs.num_kv, d)
            v_ref = full.qkv_proj(x)[:, v_rows].view(7, hs.num_kv, d)
            for j, h in enumerate(hs.kv_heads):
                w = int(plan.v_dims[0, h])
                torch.testing.assert_close(v_out[:, j, :w], v_ref[:, j] @ v_basis[h, :, :w], atol=1e-5, rtol=1e-4)
                assert torch.count_nonzero(v_out[:, j, w:]) == 0
            q_heads = range(hs.q_start, hs.q_start + hs.num_q)
            padded = torch.zeros(7, hs.num_q, d)
            padded[..., : attn.shape[-1]] = attn[:, list(q_heads)]
            got = block.o_proj(padded.view(7, -1))
            full_attn = torch.stack([attn[:, i] @ v_basis[group[i]].T for i in q_heads], dim=1)
            want = full_attn.reshape(7, -1) @ full.o_proj.weight[:, q_rows].T
            torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-4)


# ------------------------------------------------------------------------------------------------ rotary embedding
@pytest.fixture(scope="module")
def vllm_context():
    from vllm.config import VllmConfig, set_current_vllm_config

    with set_current_vllm_config(VllmConfig()):
        yield


def _rotary(head_dim: int, rotary_dim: int, neox: bool):
    from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding

    return RotaryEmbedding(head_dim, rotary_dim, 512, 10000.0, neox, torch.float32).to(DEVICE)


@pytest.mark.parametrize("neox,rotary_dim", [(True, 64), (False, 64), (True, 32)])
def test_rotary_matches_vllm_and_inverts(vllm_context, neox, rotary_dim):
    from fastdc.vllm_plugin.rope import Rotary

    module = _rotary(64, rotary_dim, neox)
    positions = torch.randint(0, 512, (9,), device=DEVICE)
    q, k = torch.randn(9, 4 * 64, device=DEVICE), torch.randn(9, 2 * 64, device=DEVICE)
    _, k_rope = module.forward_native(positions, q.clone(), k.clone())
    rotary = Rotary(module, 64)
    cos, sin = rotary.cos_sin(positions)
    got = rotary.rotate(k.view(9, 2, 64), cos[:, None], sin[:, None])
    torch.testing.assert_close(got.reshape(9, -1), k_rope)
    back = rotary.rotate(got, cos[:, None], sin[:, None], inverse=True)
    torch.testing.assert_close(back, k.view(9, 2, 64), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("neox", [True, False])
def test_palu_kernels_match_pytorch(vllm_context, neox):
    from fastdc.vllm_plugin.palu import PaluKeys
    from fastdc.vllm_plugin.rope import Rotary

    gen = torch.Generator().manual_seed(1)
    k_dims, v_dims, bs, d = (24, 64), (16, 40), 16, 64
    basis = _orthonormal(2, dim=d, generator=gen)[..., :64].clone()
    for h, w in enumerate(k_dims):
        basis[h, :, w:] = 0
    rotary = Rotary(_rotary(d, d, neox), d)
    kernels = PaluKeys(basis.to(DEVICE), rotary, torch.bfloat16, use_kernels=True)
    torch_ops = PaluKeys(basis.to(DEVICE), rotary, torch.bfloat16, use_kernels=False)
    assert kernels.use_kernels and not torch_ops.use_kernels

    k = torch.randn(37, 2, d, device=DEVICE, dtype=torch.bfloat16)
    positions = torch.randint(0, 500, (37,), device=DEVICE)
    lat_kernel, lat_torch = kernels.compress(k, positions), torch_ops.compress(k, positions)
    torch.testing.assert_close(lat_kernel[..., :64].float(), lat_torch.float(), atol=3e-2, rtol=2e-2)

    layout = layer_layout(k_dims, v_dims, bs, device=DEVICE)
    pages = torch.randn(9, layout.page_elems, device=DEVICE, dtype=torch.bfloat16)
    src = torch.tensor([3, 0, 7, 5], device=DEVICE)
    logical = torch.tensor([0, 5, 2, 31], device=DEVICE)
    offsets = layout.k.offsets.tolist()
    out_kernel = torch.zeros(4, 2 * bs * d, device=DEVICE, dtype=torch.bfloat16)
    out_torch = torch.zeros_like(out_kernel)
    kernels.decompress(pages, layout.k, k_dims, offsets, src, logical, out_kernel)
    torch_ops.decompress(pages, layout.k, k_dims, offsets, src, logical, out_torch)
    torch.testing.assert_close(out_kernel.float(), out_torch.float(), atol=5e-2, rtol=2e-2)


# ---------------------------------------------------------------------------------------- one layer, step by step
class PagedRequests:
    """Requests on a paged cache, scheduled step by step as vLLM does (decodes first, chunked prefills)."""

    def __init__(self, block_size: int, num_blocks: int, seed: int = 0):
        self.block_size = block_size
        gen = torch.Generator().manual_seed(seed)
        self.free = (torch.randperm(num_blocks - 1, generator=gen) + 1).tolist()  # block 0 is vLLM's null block
        self.blocks: dict[str, list[int]] = {}
        self.length: dict[str, int] = {}

    def common(self, work: list[tuple[str, int]], padding: int = 0):
        """CommonAttentionMetadata of one step; `work` lists (request, new tokens), decodes first."""
        from vllm.v1.attention.backend import CommonAttentionMetadata

        bs = self.block_size
        starts, seq_lens, positions, slots, tables = [0], [], [], [], []
        for req, n in work:
            blocks, old = self.blocks.setdefault(req, []), self.length.get(req, 0)
            new = old + n
            while len(blocks) * bs < new:
                blocks.append(self.free.pop())
            self.length[req] = new
            starts.append(starts[-1] + n)
            seq_lens.append(new)
            positions += list(range(old, new))
            slots += [blocks[p // bs] * bs + p % bs for p in range(old, new)]
            tables.append(blocks)
        width = max(len(t) for t in tables) + 1
        table = torch.tensor([t + [0] * (width - len(t)) for t in tables], dtype=torch.int32)
        num_tokens = starts[-1]
        qsl, lens = torch.tensor(starts, dtype=torch.int32), torch.tensor(seq_lens, dtype=torch.int32)
        return CommonAttentionMetadata(
            query_start_loc=qsl.to(DEVICE), query_start_loc_cpu=qsl, seq_lens=lens.to(DEVICE), num_reqs=len(work),
            num_actual_tokens=num_tokens, max_query_len=max(n for _, n in work), max_seq_len=max(seq_lens),
            block_table_tensor=table.to(DEVICE),
            slot_mapping=torch.tensor(slots + [-1] * padding, dtype=torch.int64, device=DEVICE),
            positions=torch.tensor(positions + [0] * padding, device=DEVICE), seq_lens_cpu_upper_bound=lens)


def _dense_reference(method, plan, rotary, scale, history, q_new):
    """Expected output of the new tokens of one request: float32 attention on all its tokens."""
    k_all = torch.cat([h[1] for h in history]).float()  # [L, Hkv, D] post-RoPE
    v_all = torch.cat([h[2] for h in history]).float()
    pos = torch.cat([h[3] for h in history])
    hkv, d = k_all.shape[1], k_all.shape[2]
    g = q_new.shape[1] // hkv
    q = q_new.float()
    v_dims = plan.v_dims[0].tolist()
    if method != "full":
        mask = torch.arange(d, device=DEVICE)[None, :] < torch.tensor(v_dims, device=DEVICE)[:, None]
        v_all = v_all * mask
    if method == "fastdc":
        r = plan.kept_k_basis(0).to(DEVICE)
        k_all = torch.einsum("lhd,hdw->lhw", k_all, r)
        q = torch.einsum("thgd,hdw->thgw", q.view(q.shape[0], hkv, g, d), r).reshape(q.shape[0], -1, r.shape[-1])
    elif method == "palu":
        b = plan.kept_k_basis(0).to(DEVICE)
        cos, sin = rotary.cos_sin(pos)
        k_pre = rotary.rotate(k_all, cos[:, None], sin[:, None], inverse=True)
        k_lat = torch.einsum("lhd,hdw->lhw", k_pre, b).bfloat16().float()  # the cache holds bfloat16
        k_all = rotary.rotate(torch.einsum("lhw,hdw->lhd", k_lat, b), cos[:, None], sin[:, None])
    n, total = q.shape[0], k_all.shape[0]
    k_rep, v_rep = k_all.repeat_interleave(g, dim=1), v_all.repeat_interleave(g, dim=1)
    scores = torch.einsum("thw,lhw->htl", q, k_rep) * scale
    causal = torch.arange(total, device=DEVICE)[None, :] > torch.arange(total - n, total, device=DEVICE)[:, None]
    probs = torch.softmax(scores.masked_fill(causal, float("-inf")), dim=-1)
    return torch.einsum("htl,lhd->thd", probs, v_rep)


@pytest.mark.parametrize("attention", ["reference", "kernels"])
@pytest.mark.parametrize("method", ["full", "fastdc", "palu"])
def test_layer_attention_follows_dense_reference(vllm_context, method, attention):
    from fastdc.vllm_plugin.rope import Rotary
    from fastdc.vllm_plugin.runtime import LayerRuntime, Runtime, resolve_ops

    torch.manual_seed(0)
    heads, d, bs, scale = 4, 64, 16, 0.125
    widths = {"full": ([64, 64], [64, 64]), "fastdc": ([40, 23], [33, 64]), "palu": ([48, 30], [17, 52])}[method]
    plan = _plan(method, *widths, num_heads=heads, head_dim=d)
    rotary = Rotary(_rotary(d, d, True), d) if method == "palu" else None
    lp = layer_plan(plan, 0, head_slice(heads, 2, 1, 0), 1)
    settings = PluginSettings(plan="unused", attention=attention, decompress_tokens=64)
    ops = resolve_ops(attention)
    layer = LayerRuntime("model.layers.0.self_attn.attn", 0, lp, rotary, torch.bfloat16, DEVICE,
                         ops.name == "kernels")
    runtime = Runtime(settings, ops, method, [layer], DEVICE, torch.bfloat16, max_model_len=64)
    cache = torch.zeros(40, 1, bs, lp.token_elems, dtype=torch.bfloat16, device=DEVICE)
    sched = PagedRequests(bs, 40)
    history: dict[str, list] = {}
    steps = [[("a", 37), ("b", 30)], [("a", 1), ("b", 21), ("c", 5)], [("a", 1), ("b", 1), ("c", 1)],
             [("c", 1), ("b", 9)]]
    for step_work in steps:
        padding = 3
        common = sched.common(step_work, padding)
        meta = runtime.step(common, bs)
        assert runtime.step(common, bs) is meta  # built once per step for all layer groups
        if method == "palu":  # a buffer of 64 keys holds the blocks of one request at a time
            assert len(meta.chunks) > 1
            assert [(c.token_start, c.token_end) for c in meta.chunks][-1][1] == common.num_actual_tokens
        num_tokens = common.num_actual_tokens + padding
        q = torch.randn(num_tokens, heads, d, device=DEVICE, dtype=torch.bfloat16)
        k = torch.randn(num_tokens, 2, d, device=DEVICE, dtype=torch.bfloat16)
        v = torch.randn(num_tokens, 2, d, device=DEVICE, dtype=torch.bfloat16)
        out = torch.full((num_tokens, heads, d), float("nan"), device=DEVICE, dtype=torch.bfloat16)
        layer.forward(runtime, meta, q, k, v, cache, out, scale)
        start = 0
        for req, n in step_work:
            positions = torch.arange(sched.length[req] - n, sched.length[req], device=DEVICE)
            history.setdefault(req, []).append((None, k[start : start + n], v[start : start + n], positions))
            want = _dense_reference(method, plan, rotary, scale, history[req], q[start : start + n])
            got = out[start : start + n].float()
            assert (got - want).norm() / want.norm() < 1e-2, (req, step_work)
            if method != "full":
                vd = torch.tensor(widths[1], device=DEVICE).repeat_interleave(heads // 2)
                assert torch.count_nonzero(got * (torch.arange(d, device=DEVICE)[None, None] >= vd[None, :, None])) == 0
            start += n
        assert torch.count_nonzero(out[start:]) == 0  # padded rows


# ------------------------------------------------------------------------------------------------ engine level
MODEL = os.environ.get("FASTDC_TEST_MODEL", "Qwen/Qwen3-0.6B")
CACHE_DIR = Path(os.environ.get("FASTDC_TEST_CACHE", Path.home() / ".cache" / "fastdc" / "tests"))
NEAR_TIE_NATS = 1.0
MAX_TOKENS = 12
PROMPTS = [
    "The capital of France is",
    "Q: What is 17 times 23? A:",
    "Write a short poem about the sea.\n",
    "Explain in two sentences why the sky is blue.",
    'def fibonacci(n):\n    """Return the n-th Fibonacci number."""\n',
    "Once upon a time, in a small village by the mountains, there lived an old clockmaker who " * 6,
]
PLAIN = {"enable_chunked_prefill": False, "enable_prefix_caching": False, "max_num_batched_tokens": 4096}
CHUNKED_CACHED = {"enable_chunked_prefill": True, "enable_prefix_caching": True, "max_num_batched_tokens": 64}

_ENGINE_SCRIPT = textwrap.dedent(
    """
    import json, sys
    from vllm import LLM, SamplingParams
    spec = json.loads(sys.argv[1])
    kwargs = dict(model=spec["model"], dtype="bfloat16", enforce_eager=True, max_model_len=4096, max_num_seqs=8,
                  gpu_memory_utilization=0.3, seed=0)
    kwargs.update(spec["engine"])
    if spec["plan"] is not None:
        kwargs["attention_config"] = {"backend": "CUSTOM"}
        kwargs["additional_config"] = {"fastdc": {"plan": spec["plan"], **spec["fastdc"]}}
    llm = LLM(**kwargs)
    params = SamplingParams(temperature=0.0, max_tokens=spec["max_tokens"], logprobs=5)
    results = []
    for _ in range(spec["passes"]):
        for out in llm.generate(spec["prompts"], params, use_tqdm=False):
            c = out.outputs[0]
            results.append({"prompt_ids": list(out.prompt_token_ids), "token_ids": list(c.token_ids),
                            "logprobs": [{str(t): lp.logprob for t, lp in step.items()} for step in c.logprobs],
                            "cached": out.num_cached_tokens or 0})
    cache = llm.llm_engine.vllm_config.cache_config
    print("RESULT" + json.dumps({"results": results, "num_blocks": cache.num_gpu_blocks}))
    """
)


def _require_plugin() -> None:
    from importlib.metadata import entry_points

    if not any(e.value.startswith("fastdc.vllm_plugin") for e in entry_points(group="vllm.general_plugins")):
        pytest.skip("the vLLM plugin is not installed: run `pip install -e .[vllm]`")


def _memory_of_other_processes() -> int:
    free, total = torch.cuda.mem_get_info(DEVICE)
    return total - free - torch.cuda.memory_reserved(DEVICE)


_baseline: list[int] = []


def _wait_for_released_memory(timeout: float = 120.0) -> None:
    """vLLM sizes its cache from the free memory; start an engine only after the previous one released its memory."""
    if not _baseline:
        _baseline.append(_memory_of_other_processes())
    deadline = time.monotonic() + timeout
    while _memory_of_other_processes() > _baseline[0] + (256 << 20) and time.monotonic() < deadline:
        time.sleep(1.0)


def _generate(plan: str | None, engine: dict | None = None, fastdc: dict | None = None, passes: int = 1) -> dict:
    """Greedy generation of PROMPTS in a fresh vLLM engine process."""
    _require_plugin()
    _wait_for_released_memory()
    spec = {"model": MODEL, "plan": plan, "engine": engine or PLAIN, "fastdc": fastdc or {}, "passes": passes,
            "prompts": PROMPTS, "max_tokens": MAX_TOKENS}
    env = dict(os.environ, VLLM_USE_FLASHINFER_SAMPLER="0")
    done = subprocess.run([sys.executable, "-c", _ENGINE_SCRIPT, json.dumps(spec)], env=env, capture_output=True,
                          text=True)
    lines = [line for line in done.stdout.splitlines() if line.startswith("RESULT")]
    assert lines, done.stderr[-4000:]
    return json.loads(lines[-1][len("RESULT"):])


def _agrees(reference: dict, tokens: list[int]) -> bool:
    """Identical up to the first position at which the reference has a near tie between the two chosen tokens."""
    for position, (want, got) in enumerate(zip(reference["token_ids"], tokens)):
        if want == got:
            continue
        top = reference["logprobs"][position]
        return str(got) in top and top[str(want)] - top[str(got)] <= NEAR_TIE_NATS
    return True


@pytest.fixture(scope="module")
def hf_model():
    from fastdc.models import load_model

    model, tokenizer = load_model(MODEL)
    yield model, tokenizer
    del model
    torch.cuda.empty_cache()


@pytest.fixture(scope="module")
def plans(hf_model, tmp_path_factory) -> dict[str, str]:
    """Plans of the test model: full, fastdc without compression, and compressed fastdc / palu plans.

    The compressed plans keep the model close to the original (K compressed in every layer, V in every fourth
    layer) so that greedy tokens stay comparable between implementations; "fastdc_dense" compresses K and V
    of every layer for the capacity test.
    """
    from fastdc.calibration import Calibration, calibrate
    from fastdc.plan import adaptive, full_plan, uniform

    path = CACHE_DIR / f"{MODEL.replace('/', '--')}-calibration.safetensors"
    if path.exists():
        calib = Calibration.load(str(path))
    else:
        model, tokenizer = hf_model
        calib = calibrate(model, tokenizer, num_positions=4096, seq_len=2048, window=4096, whiten_tokens=16384)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        calib.save(str(path))
    shape = calib.shape

    def v_in_every_fourth_layer(plan):
        plan.v_dims[[layer % 4 != 0 for layer in range(shape.num_layers)]] = shape.head_dim
        return plan

    built = {
        "full": full_plan(shape),
        "fastdc_zero": adaptive(calib, 0.0, "fastdc"),
        "fastdc": v_in_every_fourth_layer(adaptive(calib, 0.003, "fastdc")),
        "palu": v_in_every_fourth_layer(uniform(calib, 0.1, "palu")),
        "fastdc_dense": adaptive(calib, 0.05, "fastdc"),
    }
    folder = tmp_path_factory.mktemp("plans")
    paths = {}
    for name, plan in built.items():
        paths[name] = str(folder / f"{name}.safetensors")
        plan.save(paths[name])
    return paths


@pytest.fixture(scope="module")
def runs(plans) -> dict[str, dict]:
    return {
        "stock": _generate(None),
        "full": _generate(plans["full"]),
        "fastdc_zero": _generate(plans["fastdc_zero"]),
        "fastdc": _generate(plans["fastdc"]),
        "fastdc_reference": _generate(plans["fastdc"], fastdc={"attention": "reference"}),
        "palu": _generate(plans["palu"]),
        "fastdc_dense": _generate(plans["fastdc_dense"]),
    }


def test_engine_full_plan_follows_default_backend(runs):
    stock, full = runs["stock"]["results"], runs["full"]["results"]
    assert all(a["token_ids"][0] == b["token_ids"][0] for a, b in zip(stock, full))
    assert all(_agrees(a, b["token_ids"]) for a, b in zip(stock, full))


def test_engine_uncompressed_fastdc_follows_full_plan(runs):
    full, zero = runs["full"]["results"], runs["fastdc_zero"]["results"]
    assert all(_agrees(a, b["token_ids"]) for a, b in zip(full, zero))


@pytest.mark.parametrize("name", ["fastdc", "fastdc_reference", "palu"])
def test_engine_compressed_plans_follow_reference_engine(runs, plans, hf_model, name):
    """Teacher-forced on the vLLM tokens, the reference engine must rank every vLLM token first or within a near
    tie for the first tokens of most prompts, and give log-probabilities close to vLLM's."""
    from fastdc.engine import Engine
    from fastdc.plan import CompressionPlan

    model, _ = hf_model
    engine = Engine(model, CompressionPlan.load(plans[name.removesuffix("_reference")]))
    following, deviations = 0, []
    for result in runs[name]["results"]:
        logits = engine.prefill(torch.tensor(result["prompt_ids"]))
        gaps = []
        for position, token in enumerate(result["token_ids"]):
            logprobs = torch.log_softmax(logits.float(), dim=-1)
            gaps.append(float(logprobs.max() - logprobs[token]))
            deviations += [abs(v - float(logprobs[int(t)])) for t, v in result["logprobs"][position].items()]
            logits = engine.decode(token)
        following += all(gap <= NEAR_TIE_NATS for gap in gaps[:8])
    assert following >= len(PROMPTS) - 1
    assert float(torch.tensor(deviations).median()) < 0.15


def test_engine_cache_capacity_grows_with_compression(runs, plans):
    def elements_per_token(name):
        plan = CompressionPlan.load(plans[name])
        return int(plan.k_dims.sum() + plan.v_dims.sum())

    full_blocks, full_elems = runs["full"]["num_blocks"], elements_per_token("full")
    assert runs["stock"]["num_blocks"] == full_blocks
    for name in ("fastdc", "fastdc_dense"):
        expected = full_elems / elements_per_token(name)
        assert runs[name]["num_blocks"] / full_blocks >= 0.97 * expected, name
    assert runs["fastdc_dense"]["num_blocks"] > 1.5 * full_blocks
    assert runs["palu"]["num_blocks"] > full_blocks  # the decompression buffer takes part of the memory


@pytest.mark.parametrize("name", ["fastdc", "palu"])
def test_engine_chunked_prefill_and_prefix_caching(runs, plans, name):
    """Prefill in chunks of 64 tokens, then run the same prompts again on the prefix cache."""
    plain = runs[name]["results"]
    chunked = _generate(plans[name], engine=CHUNKED_CACHED, passes=2)["results"]
    first, second = chunked[: len(PROMPTS)], chunked[len(PROMPTS):]
    assert max(len(r["prompt_ids"]) for r in plain) > CHUNKED_CACHED["max_num_batched_tokens"]
    assert max(r["cached"] for r in second) > CHUNKED_CACHED["max_num_batched_tokens"]
    assert all(_agrees(a, b["token_ids"]) for a, b in zip(plain, first))
    assert all(_agrees(a, b["token_ids"]) for a, b in zip(plain, second))


def test_engine_piecewise_cuda_graphs(runs, plans):
    graphs = _generate(plans["fastdc"], engine={**PLAIN, "enforce_eager": False})["results"]
    assert all(_agrees(a, b["token_ids"]) for a, b in zip(runs["fastdc"]["results"], graphs))
