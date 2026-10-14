"""vLLM attention backend of FastDC, registered as `AttentionBackendEnum.CUSTOM`.

Select it with `--attention-backend CUSTOM` and give the compression plan through the "fastdc" entry of
`--additional-config` (see `settings.py`); offline: `LLM(model, attention_backend="CUSTOM",
additional_config={"fastdc": {"plan": "plan.safetensors"}})`.

The cache page of a layer holds `block_size` tokens of every KV head of the rank at the layer's own
widths (`fastdc.kernels.layout`); vLLM sizes the pages from the per-layer spec that `loader.py` publishes,
so the number of cache blocks grows with the compression. The backend writes the cache itself
(`forward_includes_kv_cache_update`). Attention runs outside CUDA graphs (AttentionCGSupport.NEVER), and vLLM
captures the rest of the model piecewise.
"""

from __future__ import annotations

from typing import ClassVar

import torch
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionLayer,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheLayout

from fastdc.vllm_plugin.runtime import LayerRuntime, StepMetadata, get_runtime

BACKEND_NAME = "CUSTOM"
BLOCK_SIZE_MULTIPLE = 16


class FastDCAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True
    forward_includes_kv_cache_update: bool = True
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[str]] = ["auto", "float16", "bfloat16"]

    @staticmethod
    def get_name() -> str:
        return BACKEND_NAME

    @staticmethod
    def get_impl_cls() -> type[FastDCAttentionImpl]:
        return FastDCAttentionImpl

    @staticmethod
    def get_builder_cls() -> type[FastDCMetadataBuilder]:
        return FastDCMetadataBuilder

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(BLOCK_SIZE_MULTIPLE)]

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        return head_size % 16 == 0 and head_size <= 256

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type == AttentionType.DECODER

    @classmethod
    def supported_kv_cache_layouts(cls) -> tuple[KVCacheLayout, ...]:
        return (KVCacheLayout.LBHNC, KVCacheLayout.LBNHC)

    @classmethod
    def customize_spec(cls, spec: AttentionSpec) -> AttentionSpec:
        # Each layer publishes its packed page itself (loader.publish_page); vLLM passes no layer here.
        return spec


class FastDCMetadataBuilder(AttentionMetadataBuilder[StepMetadata]):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.NEVER

    def __init__(self, kv_cache_spec: AttentionSpec, layer_names: list[str], vllm_config, device: torch.device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self._init_reorder_batch_threshold(1, supports_spec_as_decode=False)
        if vllm_config.speculative_config is not None:
            raise ValueError("speculative decoding is not supported by the FastDC backend")
        self.runtime = get_runtime()
        self.block_size = kv_cache_spec.block_size

    def build(self, common_prefix_len: int, common_attn_metadata: CommonAttentionMetadata,
              fast_build: bool = False) -> StepMetadata:
        return self.runtime.step(common_attn_metadata, self.block_size)


class FastDCAttentionImpl(AttentionImpl[StepMetadata]):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        sliding_window: int | None = None,
        kv_cache_dtype: str = "auto",
        logits_soft_cap: float | None = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        **kwargs,
    ) -> None:
        if alibi_slopes is not None or sliding_window is not None or logits_soft_cap:
            raise ValueError("ALiBi, sliding-window attention and logit soft-capping are not supported by FastDC")
        if kv_sharing_target_layer_name is not None or kwargs.get("sinks") is not None:
            raise ValueError("KV sharing across layers and attention sinks are not supported by FastDC")
        if attn_type != AttentionType.DECODER:
            raise ValueError(f"attention type {attn_type} is not supported by FastDC")
        if self.dcp_world_size > 1 or self.pcp_world_size > 1:
            raise ValueError("context parallelism is not supported by FastDC")
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads or num_heads
        self.kv_cache_dtype = kv_cache_dtype
        self._layer: LayerRuntime | None = None

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: StepMetadata | None,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError("fused output quantization is not supported by FastDC")
        if attn_metadata is None:  # memory profiling run without a cache
            return output.fill_(0)
        runtime = get_runtime()
        if self._layer is None:
            self._layer = runtime.layer(layer.layer_name)
        self._layer.forward(runtime, attn_metadata, query, key, value, kv_cache, output, self.scale)
        return output
