"""Preparation of a loaded model for the FastDC backend.

After vLLM has loaded and post-processed the weights of a model whose attention layers use the FastDC
backend, `prepare_model`
  - loads the compression plan and checks it against the model,
  - folds the V basis of every layer into its qkv_proj and o_proj weights (fastdc and palu),
  - publishes every layer's cache page size (K and V widths of its heads) to vLLM, and
  - creates the runtime that the attention layers and metadata builders of this worker share.
"""

from __future__ import annotations

import functools
from dataclasses import replace

import torch
from torch import nn
from vllm.logger import init_logger

from fastdc.plan import CompressionPlan
from fastdc.vllm_plugin.layers import check_projections, fold_v_basis, head_slice, layer_plan
from fastdc.vllm_plugin.rope import Rotary
from fastdc.vllm_plugin.runtime import LayerRuntime, Runtime, resolve_ops, set_runtime
from fastdc.vllm_plugin.settings import settings_for

logger = init_logger(f"vllm.{__name__}")


def install_load_hook() -> None:
    """Run `prepare_model` on every model that a vLLM model loader returns in this process."""
    from vllm.model_executor.model_loader.base_loader import BaseModelLoader

    load_model = BaseModelLoader.load_model
    if getattr(load_model, "_fastdc_hook", False):
        return

    @functools.wraps(load_model)
    def load_and_prepare(self, vllm_config, model_config, prefix: str = ""):
        model = load_model(self, vllm_config, model_config, prefix)
        prepare_model(model, vllm_config)
        return model

    load_and_prepare._fastdc_hook = True
    BaseModelLoader.load_model = load_and_prepare


def publish_page(attn: nn.Module, token_elems: int) -> None:
    """Make vLLM size the cache pages of this layer for `token_elems` cache elements per token.

    vLLM asks every attention layer for its KV cache spec (`Attention.get_kv_cache_spec`) and lets the backend
    adjust it without naming the layer (`AttentionBackend.customize_spec`), so the per-layer page is published
    by the layer: one head slot per token holding the K and V widths of all its KV heads.
    """
    from vllm.utils.torch_utils import get_dtype_size

    vllm_spec = attn.get_kv_cache_spec

    def get_kv_cache_spec(vllm_config):
        spec = vllm_spec(vllm_config)
        return replace(spec, num_head_slots=1, state_content_bytes=token_elems * get_dtype_size(spec.dtype))

    attn.get_kv_cache_spec = get_kv_cache_spec


def _uses_fastdc(module: nn.Module) -> bool:
    from vllm.model_executor.layers.attention import Attention

    from fastdc.vllm_plugin.backend import FastDCAttentionBackend

    return isinstance(module, Attention) and issubclass(module.get_attn_backend(), FastDCAttentionBackend)


def _fastdc_layers(model: nn.Module) -> list[tuple[str, nn.Module, nn.Module]]:
    """(layer name, attention block, vLLM Attention) of every attention layer that uses the FastDC backend."""
    found = [(m.attn.layer_name, m, m.attn) for m in model.modules() if _uses_fastdc(getattr(m, "attn", None))]
    blocks = {attn.layer_name for _, _, attn in found}
    others = [name for name, m in model.named_modules() if _uses_fastdc(m) and m.layer_name not in blocks]
    if others:
        raise ValueError(f"attention layers such as {others[0]} are not held as `attn` by an attention block with "
                         "qkv_proj and o_proj; this model is not supported by the FastDC backend")
    return found


def _check_plan(plan: CompressionPlan, vllm_config) -> None:
    hf = vllm_config.model_config.hf_text_config
    shape = plan.shape
    model = (hf.num_hidden_layers, hf.num_attention_heads,
             getattr(hf, "num_key_value_heads", None) or hf.num_attention_heads,
             vllm_config.model_config.get_head_size())
    planned = (shape.num_layers, shape.num_heads, shape.num_kv_heads, shape.head_dim)
    if model != planned:
        raise ValueError(f"the plan was made for (layers, heads, KV heads, head dim) = {planned}, "
                         f"the model has {model}")


def _check_engine(vllm_config) -> None:
    if vllm_config.speculative_config is not None:
        raise ValueError("speculative decoding is not supported by the FastDC backend")
    if vllm_config.lora_config is not None:
        raise ValueError("LoRA adapters are not supported by the FastDC backend")
    cache_dtype = vllm_config.cache_config.cache_dtype
    model_dtype = vllm_config.model_config.dtype
    if cache_dtype != "auto" and cache_dtype != str(model_dtype).removeprefix("torch."):
        raise ValueError("the FastDC backend stores the cache in the model dtype; use --kv-cache-dtype auto")


@torch.no_grad()
def prepare_model(model: nn.Module, vllm_config) -> None:
    """Apply the compression plan to a loaded model whose attention layers use the FastDC backend."""
    layers = _fastdc_layers(model)
    if not layers:
        return
    from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size
    from vllm.model_executor.models.utils import extract_layer_index

    _check_engine(vllm_config)
    settings = settings_for(vllm_config)
    plan = CompressionPlan.load(settings.plan)
    _check_plan(plan, vllm_config)
    tp_size, tp_rank = get_tensor_model_parallel_world_size(), get_tensor_model_parallel_rank()
    heads = head_slice(plan.shape.num_heads, plan.shape.num_kv_heads, tp_size, tp_rank)
    dtype = vllm_config.model_config.dtype
    ops = resolve_ops(settings.attention)

    runtimes: list[LayerRuntime] = []
    for name, block, attn in layers:
        if (attn.num_heads, attn.num_kv_heads, attn.head_size) != (heads.num_q, heads.num_kv, plan.shape.head_dim):
            raise ValueError(f"{name} has {attn.num_heads} query and {attn.num_kv_heads} KV heads of size "
                             f"{attn.head_size} on this rank; the plan gives {heads.num_q}, {heads.num_kv}, "
                             f"{plan.shape.head_dim}")
        lp = layer_plan(plan, extract_layer_index(name), heads, tp_size)
        check_projections(block, lp)
        if plan.method != "full" and not getattr(block, "_fastdc_folded", False):
            fold_v_basis(block, lp)
            block._fastdc_folded = True
        rotary = None
        if plan.method == "palu":
            if getattr(block, "rotary_emb", None) is None:
                raise ValueError(f"{type(block).__name__} has no rotary_emb; the palu method needs it")
            rotary = Rotary(block.rotary_emb, lp.head_dim)
        device = block.qkv_proj.weight.device
        runtimes.append(LayerRuntime(name, len(runtimes), lp, rotary, dtype, device, ops.name == "kernels"))
        publish_page(attn, lp.token_elems)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # release the temporaries of the weight folding before vLLM measures memory

    set_runtime(Runtime(settings, ops, plan.method, runtimes, device, dtype, vllm_config.model_config.max_model_len,
                        profile=settings.profile if tp_rank == 0 else ""))
    stored = sum(rt.plan.token_elems for rt in runtimes)
    full = 2 * heads.num_kv * plan.shape.head_dim * len(runtimes)
    logger.info("FastDC: %s plan %s applied to %d attention layers (TP rank %d of %d); cache elements per token "
                "%d of %d (%.1f%%); attention: %s", plan.method, settings.plan, len(runtimes), tp_rank, tp_size,
                stored, full, 100.0 * stored / full, ops.name)
