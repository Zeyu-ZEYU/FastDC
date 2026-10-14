"""State of the FastDC backend in one worker process: the attention layers, the kernels and the step metadata.

Per step, vLLM builds attention metadata once per group of layers with equal cache pages; layers with
different widths form different groups, so the step metadata, including the kernel schedules of all
layers, is built for the first group and reused by the others (`Runtime.step`).

Forward of one layer (q, k post-RoPE [T, H, D] / [T, Hkv, D]; v [T, Hkv, D]):
    full    stores k and v at full width; attention on q, k, v
    fastdc  q_c = q R and k_c = k R per KV group (R: kept rotation of the KV head); v already holds
            V' = E A_V in its first features (V basis folded into qkv_proj); stores k_c and v; attention
            on q_c with the original scale; the output stays in V' coordinates (folded into o_proj)
    palu    k_lat = RoPE^-1(k) B_K; stores k_lat and v; before attention, the keys of every block the
            step reads are reconstructed, K = RoPE(k_lat B_K^T), into a full-width buffer that the
            attention reads through a remapped block table (fastdc.vllm_plugin.palu)
"""

from __future__ import annotations

import contextlib
import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from vllm.logger import init_logger
from vllm.v1.attention.backend import AttentionMetadata

from fastdc.kernels.layout import LayerLayout, TileLayout, layer_layout
from fastdc.vllm_plugin.layers import LayerPlan
from fastdc.vllm_plugin.palu import PaluKeys
from fastdc.vllm_plugin.rope import Rotary
from fastdc.vllm_plugin.settings import PluginSettings

logger = init_logger(f"vllm.{__name__}")

DEFAULT_DECOMPRESS_TOKENS = 65536


# ------------------------------------------------------------------------------------------------ kernels
@dataclass(frozen=True)
class AttentionOps:
    name: str
    write_kv: Callable[..., None]
    paged_attention: Callable[..., torch.Tensor]
    build_schedules: Callable[..., list] | None = None
    schedule_workspace: Callable[[], Any] | None = None  # reusable output buffers of build_schedules


def resolve_ops(attention: str) -> AttentionOps:
    """The Triton kernels of `fastdc.kernels` ("kernels") or the PyTorch reference ("reference")."""
    from fastdc.kernels import reference

    if attention == "kernels":
        import fastdc.kernels as kernels

        names = ("write_kv", "paged_attention", "build_schedules")
        missing = [n for n in names if not hasattr(kernels, n)]
        if not missing:
            return AttentionOps("kernels", kernels.write_kv, kernels.paged_attention, kernels.build_schedules,
                                getattr(kernels, "ScheduleWorkspace", None))
        logger.warning("fastdc.kernels does not provide %s; using the PyTorch reference attention", ", ".join(missing))
    return AttentionOps("reference", reference.write_kv, reference.paged_attention)


# ------------------------------------------------------------------------------------------------ profiling
class StepProfiler:
    """Per-step GPU time of the FastDC operations, measured with CUDA events and written as JSON lines.

    Each line holds the wall-clock time at which the step's events were collected, the step size, and the
    seconds spent in "compress" (Q/K rotation, or the key projection of palu), "schedule" (kernel
    schedules) and "decompress" (palu key reconstruction). Collecting a step waits for its last event.
    """

    KINDS = ("compress", "schedule", "decompress")

    def __init__(self, path: str):
        self.path = path
        self.enabled = bool(path)
        self._events: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []
        self._step: dict = {}
        if self.enabled:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    @contextlib.contextmanager
    def region(self, kind: str):
        if not self.enabled:
            yield
            return
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        yield
        end.record()
        self._events.append((kind, start, end))

    def begin_step(self, num_reqs: int, num_tokens: int) -> None:
        if not self.enabled:
            return
        self.flush()
        self._step = {"num_reqs": num_reqs, "num_tokens": num_tokens}

    def flush(self) -> None:
        if not self._events:
            return
        self._events[-1][2].synchronize()
        totals = dict.fromkeys(self.KINDS, 0.0)
        for kind, start, end in self._events:
            totals[kind] += start.elapsed_time(end) / 1000.0
        line = {"time": time.time(), **self._step, **{f"{k}_s": v for k, v in totals.items()}}
        with open(self.path, "a") as f:
            f.write(json.dumps(line) + "\n")
        self._events = []


# ------------------------------------------------------------------------------------------------ step metadata
@dataclass
class DecompressChunk:
    """A run of consecutive requests of a step whose keys fit the decompression buffer at once (palu)."""

    token_start: int
    token_end: int
    query_start_loc: torch.Tensor  # [n + 1] int32, relative to token_start
    seq_lens: torch.Tensor  # [n] int32
    block_table: torch.Tensor  # [n, max_blocks] int32
    k_block_table: torch.Tensor  # [n, max_blocks] int32: blocks of the decompression buffer
    src_blocks: torch.Tensor  # [m] int64: buffer block i holds the keys of cache block src_blocks[i] ...
    logical_blocks: torch.Tensor  # [m] int64: ... which is block logical_blocks[i] of its request
    schedules: list | None = None


@dataclass
class StepMetadata(AttentionMetadata):
    """Attention metadata of one step, shared by all layers."""

    num_actual_tokens: int
    num_reqs: int
    num_decodes: int  # requests [0, num_decodes) have one query token each
    max_query_len: int
    max_seq_len: int
    query_start_loc: torch.Tensor  # [B + 1] int32
    seq_lens: torch.Tensor  # [B] int32
    block_table: torch.Tensor  # [B, max_blocks] int32
    slot_mapping: torch.Tensor  # [T] int64
    seq_lens_cpu: np.ndarray
    positions: torch.Tensor | None = None  # [T] int64 (palu)
    chunks: list[DecompressChunk] = field(default_factory=list)  # palu
    schedules: list | None = None  # one per layer, in Runtime.layers order (kernels)


def _upload(values: np.ndarray, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    host = torch.from_numpy(np.ascontiguousarray(values)).to(dtype)
    if device.type == "cuda":
        host = host.pin_memory()
    return host.to(device, non_blocking=True)


# ------------------------------------------------------------------------------------------------ layers
class LayerRuntime:
    """One attention layer: its plan slice, projection bases, cache layout and the page view of its cache."""

    def __init__(self, name: str, position: int, plan: LayerPlan, rotary: Rotary | None, dtype: torch.dtype,
                 device: torch.device, use_kernels: bool):
        if plan.method == "palu" and rotary is None:
            raise ValueError("the palu method needs the rotary embedding of every attention layer")
        self.name = name
        self.position = position  # index of the layer's schedule
        self.plan = plan
        self.dtype = dtype
        self.device = device
        self.basis = self.q_basis = self.palu = None
        if plan.method == "fastdc":
            # [Hkv, D, Wk] kept rotation R of every KV head, zero-padded to a multiple of 8 columns for the GEMMs
            width = -(-plan.k_basis.shape[-1] // 8) * 8
            self.basis = torch.nn.functional.pad(plan.k_basis, (0, width - plan.k_basis.shape[-1])).to(device, dtype)
            self.q_basis = self.basis.repeat_interleave(plan.group_size, dim=0)  # [H, D, Wk]: R of every query head
        elif plan.method == "palu":
            self.palu = PaluKeys(plan.k_basis.to(device), rotary, dtype, use_kernels)
        self.layout: LayerLayout | None = None
        self.pages: torch.Tensor | None = None
        self._k_offsets: list[int] = []
        self._bound: tuple[int, tuple[int, ...]] | None = None

    def set_block_size(self, block_size: int) -> None:
        """Create the cache layout for pages of `block_size` tokens (once; the layout objects stay fixed)."""
        if self.layout is not None:
            if self.layout.block_size != block_size:
                raise RuntimeError(f"{self.name}: block size changed from {self.layout.block_size} to {block_size}")
            return
        host = layer_layout(self.plan.k_dims, self.plan.v_dims, block_size)
        page = block_size * self.plan.token_elems
        if host.page_elems > page:
            raise RuntimeError(f"a page of {self.name} holds {page} elements, the layout needs {host.page_elems}")
        self.layout = LayerLayout(*(TileLayout(t.dims.to(self.device), t.offsets.to(self.device), page, block_size)
                                    for t in (host.k, host.v)))
        self._k_offsets = host.k.offsets.tolist()

    def bind(self, kv_cache: torch.Tensor) -> torch.Tensor:
        """View vLLM's cache tensor of this layer, [blocks, 1, block_size, elements per token], as pages."""
        key = (kv_cache.data_ptr(), tuple(kv_cache.shape))
        if key == self._bound:
            return self.pages
        if kv_cache.dim() != 4 or kv_cache.shape[1] != 1 or kv_cache.shape[3] != self.plan.token_elems:
            raise RuntimeError(
                f"the cache of {self.name} has shape {tuple(kv_cache.shape)}, expected [blocks, 1, block_size, "
                f"{self.plan.token_elems}]: the compression plan was not applied when the model was loaded")
        num_blocks, _, block_size, per_token = kv_cache.shape
        self.set_block_size(block_size)
        page = self.layout.page_elems
        if kv_cache.stride(0) != page or kv_cache.stride(2) != per_token or kv_cache.stride(3) != 1:
            raise RuntimeError(f"the cache of {self.name} is not stored as dense pages; "
                               "use a layer-major KV cache layout (VLLM_KV_CACHE_LAYOUT=LBHNC)")
        self.pages = torch.as_strided(kv_cache, (num_blocks, page), (page, 1))
        self._bound = key
        return self.pages

    # -------------------------------------------------------------------------------------------- forward
    def forward(self, runtime: Runtime, step: StepMetadata, query: torch.Tensor, key: torch.Tensor,
                value: torch.Tensor, kv_cache: torch.Tensor, output: torch.Tensor, scale: float) -> None:
        pages = self.bind(kv_cache)
        n = step.num_actual_tokens
        q, k, v, out = query[:n], key[:n], value[:n], output[:n]
        if n:
            if self.plan.method == "palu":
                self._forward_palu(runtime, step, q, k, v, pages, out, scale)
            else:
                if self.plan.method == "fastdc":
                    with runtime.profiler.region("compress"):
                        q, k = self.rotate_queries(q), self.rotate_keys(k)
                runtime.ops.write_kv(pages, self.layout, k, v, step.slot_mapping)
                runtime.attention(q, pages, self.layout.k, pages, self.layout.v, step.block_table,
                                  step.query_start_loc, step.seq_lens, scale, out,
                                  schedule=None if step.schedules is None else step.schedules[self.position])
        if output.shape[0] > n:
            output[n:].zero_()

    def rotate_queries(self, q: torch.Tensor) -> torch.Tensor:
        """fastdc: q [T, H, D] -> q R of each query head's KV group, [T, H, Wk]."""
        return self._project(q, self.q_basis)

    def rotate_keys(self, k: torch.Tensor) -> torch.Tensor:
        """fastdc: k [T, Hkv, D] -> k R, [T, Hkv, Wk]."""
        return self._project(k, self.basis)

    @staticmethod
    def _project(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
        """x [T, N, D] @ basis [N, D, W] per head, as one batched GEMM into a contiguous [T, N, W] result."""
        out = torch.empty(x.shape[0], basis.shape[0], basis.shape[2], dtype=x.dtype, device=x.device)
        torch.bmm(x.transpose(0, 1), basis, out=out.transpose(0, 1))
        return out

    def _forward_palu(self, runtime: Runtime, step: StepMetadata, q, k, v, pages, out, scale) -> None:
        with runtime.profiler.region("compress"):
            k_lat = self.palu.compress(k, step.positions)
        runtime.ops.write_kv(pages, self.layout, k_lat, v, step.slot_mapping)
        buffer, tiles = runtime.key_buffer()
        for chunk in step.chunks:
            with runtime.profiler.region("decompress"):
                self.palu.decompress(pages, self.layout.k, self.plan.k_dims, self._k_offsets, chunk.src_blocks,
                                     chunk.logical_blocks, buffer)
            rows = slice(chunk.token_start, chunk.token_end)
            runtime.attention(q[rows], buffer, tiles, pages, self.layout.v, chunk.block_table, chunk.query_start_loc,
                              chunk.seq_lens, scale, out[rows], k_block_table=chunk.k_block_table,
                              schedule=None if chunk.schedules is None else chunk.schedules[self.position])


# ------------------------------------------------------------------------------------------------ runtime
class Runtime:
    def __init__(self, settings: PluginSettings, ops: AttentionOps, method: str, layers: list[LayerRuntime],
                 device: torch.device, dtype: torch.dtype, max_model_len: int, profile: str | None = None):
        self.settings = settings
        self.ops = ops
        self.method = method
        self.layers = {layer.name: layer for layer in layers}
        self.order = layers
        self.device = device
        self.dtype = dtype
        self.max_model_len = max_model_len
        self.block_size: int | None = None
        self._schedule_layouts: list[LayerLayout] = []
        self._key_tiles: TileLayout | None = None
        self._workspaces: list[Any] = []  # one per schedule set that is alive at a time (palu: per chunk)
        self._last_common: Any = None
        self._last_step: StepMetadata | None = None
        self.decompress_buffer: torch.Tensor | None = None
        # with tensor parallelism only the first rank records (the ranks run the same steps)
        self.profiler = StepProfiler(settings.profile if profile is None else profile)
        if method == "palu":
            # allocated with the model, before vLLM sizes the KV cache from the free memory
            first = layers[0].plan
            tokens = max(settings.decompress_tokens or max(DEFAULT_DECOMPRESS_TOKENS, max_model_len), max_model_len)
            self.decompress_buffer = torch.empty((tokens, first.heads.num_kv * first.head_dim), dtype=dtype,
                                                 device=device)

    def layer(self, name: str) -> LayerRuntime:
        try:
            return self.layers[name]
        except KeyError:
            raise RuntimeError(f"attention layer {name} was not prepared by FastDC") from None

    def _set_block_size(self, block_size: int) -> None:
        if self.block_size == block_size:
            return
        for layer in self.order:
            layer.set_block_size(block_size)
        self.block_size = block_size
        if self.method == "palu":
            plan = self.order[0].plan
            num_kv, d = plan.heads.num_kv, plan.head_dim
            dims = torch.full((num_kv,), d, dtype=torch.int32, device=self.device)
            offsets = torch.arange(num_kv, dtype=torch.int64, device=self.device) * block_size * d
            self._key_tiles = TileLayout(dims, offsets, num_kv * block_size * d, block_size)
            self._schedule_layouts = [LayerLayout(self._key_tiles, layer.layout.v) for layer in self.order]
        else:
            self._schedule_layouts = [layer.layout for layer in self.order]

    def key_buffer(self) -> tuple[torch.Tensor, TileLayout]:
        """palu: the decompression buffer as full-width key pages, and their tile layout."""
        page = self._key_tiles.page_elems
        flat = self.decompress_buffer.view(-1)
        return flat[: flat.numel() // page * page].view(-1, page), self._key_tiles

    def attention(self, q, k_pages, k_tiles, v_pages, v_tiles, block_table, query_start_loc, seq_lens, scale, out,
                  k_block_table=None, schedule=None) -> None:
        kwargs = {} if k_block_table is None else {"k_block_table": k_block_table}
        if self.ops.build_schedules is not None:
            kwargs.update(schedule=schedule, balance=self.settings.balance)
        self.ops.paged_attention(q, k_pages, k_tiles, v_pages, v_tiles, block_table, query_start_loc, seq_lens,
                                 scale, out, **kwargs)

    # -------------------------------------------------------------------------------------------- per step
    def step(self, common, block_size: int) -> StepMetadata:
        """Metadata of the step described by vLLM's CommonAttentionMetadata (built once per step)."""
        if common is self._last_common and self._last_step is not None:
            return self._last_step
        self._set_block_size(block_size)
        step = self._build_step(common)
        self._last_common, self._last_step = common, step
        return step

    def _build_step(self, common) -> StepMetadata:
        num_reqs, num_tokens = common.num_reqs, common.num_actual_tokens
        self.profiler.begin_step(num_reqs, num_tokens)
        qsl_cpu = common.query_start_loc_cpu[: num_reqs + 1].numpy().astype(np.int64)
        seq_cpu = common.seq_lens_cpu_upper_bound
        seq_cpu = (seq_cpu if seq_cpu is not None else common.seq_lens.cpu())[:num_reqs].numpy().astype(np.int64)
        q_lens = np.diff(qsl_cpu)
        not_decode = np.flatnonzero(q_lens != 1)
        step = StepMetadata(
            num_actual_tokens=num_tokens, num_reqs=num_reqs,
            num_decodes=int(not_decode[0]) if len(not_decode) else num_reqs,
            max_query_len=int(q_lens.max(initial=0)), max_seq_len=int(seq_cpu.max(initial=0)),
            query_start_loc=common.query_start_loc[: num_reqs + 1], seq_lens=common.seq_lens[:num_reqs],
            block_table=common.block_table_tensor[:num_reqs], slot_mapping=common.slot_mapping[:num_tokens],
            seq_lens_cpu=seq_cpu)
        if self.method == "palu":
            step.positions = self._positions(common, qsl_cpu, seq_cpu, num_tokens)
            step.chunks = self._decompress_chunks(step, qsl_cpu)
        elif self.ops.build_schedules is not None and num_tokens:
            step.schedules = self._schedules(step.query_start_loc, step.seq_lens, num_tokens, step.max_query_len)
        return step

    def _schedules(self, query_start_loc, seq_lens, num_tokens: int, max_query_len: int, slot: int = 0) -> list:
        """Kernel schedules of all layers for one batch; `slot` selects the reused output buffers."""
        kwargs = {}
        if self.ops.schedule_workspace is not None:
            while len(self._workspaces) <= slot:
                self._workspaces.append(self.ops.schedule_workspace())
            kwargs["workspace"] = self._workspaces[slot]
        num_heads = self.order[0].plan.heads.num_q
        with self.profiler.region("schedule"):
            return self.ops.build_schedules(self._schedule_layouts, query_start_loc, seq_lens, num_heads,
                                            balance=self.settings.balance, num_tokens=num_tokens,
                                            max_query_len=max_query_len, **kwargs)

    def _positions(self, common, qsl_cpu: np.ndarray, seq_cpu: np.ndarray, num_tokens: int) -> torch.Tensor:
        if common.positions is not None:
            return common.positions[:num_tokens].long()
        q_lens = np.diff(qsl_cpu)
        req = np.repeat(np.arange(len(q_lens)), q_lens)
        pos = np.arange(num_tokens) - qsl_cpu[req] + (seq_cpu - q_lens)[req]
        return _upload(pos, self.device, torch.int64)

    def _decompress_chunks(self, step: StepMetadata, qsl_cpu: np.ndarray) -> list[DecompressChunk]:
        """palu: split the requests into runs whose key blocks fit the decompression buffer."""
        capacity = self.decompress_buffer.shape[0] // self.block_size
        blocks = -(-step.seq_lens_cpu // self.block_size)
        chunks, start = [], 0
        while start < step.num_reqs:
            end, total = start, 0
            while end < step.num_reqs and (end == start or total + blocks[end] <= capacity):
                total += int(blocks[end])
                end += 1
            if total > capacity:
                raise RuntimeError(f"a request needs {total} blocks of decompressed keys and the buffer holds "
                                   f"{capacity}; raise the decompress_tokens setting")
            chunks.append(self._chunk(step, qsl_cpu, start, end, blocks[start:end], len(chunks)))
            start = end
        return chunks

    def _chunk(self, step: StepMetadata, qsl_cpu: np.ndarray, start: int, end: int, blocks: np.ndarray,
               index: int) -> DecompressChunk:
        n, total = end - start, int(blocks.sum())
        req = np.repeat(np.arange(n), blocks)
        logical = np.arange(total) - np.repeat(np.cumsum(blocks) - blocks, blocks)
        k_table = np.zeros((n, step.block_table.shape[1]), dtype=np.int32)
        k_table[req, logical] = np.arange(total, dtype=np.int32)
        req_gpu = _upload(req + start, self.device, torch.int64)
        logical_gpu = _upload(logical, self.device, torch.int64)
        qsl = qsl_cpu[start : end + 1]
        if n == step.num_reqs:
            query_start_loc = step.query_start_loc
        else:
            query_start_loc = step.query_start_loc[start : end + 1] - int(qsl[0])
        chunk = DecompressChunk(
            token_start=int(qsl[0]), token_end=int(qsl[-1]), query_start_loc=query_start_loc,
            seq_lens=step.seq_lens[start:end], block_table=step.block_table[start:end],
            k_block_table=_upload(k_table, self.device, torch.int32),
            src_blocks=step.block_table[req_gpu, logical_gpu].long(), logical_blocks=logical_gpu)
        if self.ops.build_schedules is not None and qsl[-1] > qsl[0]:
            chunk.schedules = self._schedules(chunk.query_start_loc, chunk.seq_lens, int(qsl[-1] - qsl[0]),
                                              int(np.diff(qsl).max()), slot=index)
        return chunk


_runtime: Runtime | None = None


def set_runtime(runtime: Runtime | None) -> None:
    global _runtime
    _runtime = runtime


def get_runtime() -> Runtime:
    """The runtime of this worker, created when the model was loaded (`loader.prepare_model`)."""
    if _runtime is None:
        raise RuntimeError("the FastDC backend was not initialized: the model must be loaded by a vLLM model "
                           "loader derived from BaseModelLoader after the fastdc plugin is registered")
    return _runtime
