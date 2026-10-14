"""Settings of the vLLM integration.

Every option is read from the key "fastdc" of vLLM's `--additional-config` and otherwise from the
environment variable FASTDC_<OPTION>:

    option             environment variable         default
    plan               FASTDC_PLAN                  (required) CompressionPlan file (.safetensors)
    attention          FASTDC_ATTENTION             kernels    (kernels | reference)
    balance            FASTDC_BALANCE               true       load-balanced kernel schedules
    decompress_tokens  FASTDC_DECOMPRESS_TOKENS     0          palu: tokens of decompressed keys held at
                                                               once (0: max(65536, max_model_len))
    profile            FASTDC_PROFILE               (none)     JSON-lines file; one line per engine step with
                                                               the GPU time of the Q/K compression, the kernel
                                                               scheduling and the key reconstruction (palu)

Example: `vllm serve Qwen/Qwen3-0.6B --attention-backend CUSTOM
          --additional-config '{"fastdc": {"plan": "plan.safetensors"}}'`
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

ADDITIONAL_CONFIG_KEY = "fastdc"
ENV_PREFIX = "FASTDC_"
ATTENTION_CHOICES = ("kernels", "reference")
_TRUE = ("1", "true", "yes", "on")
_FALSE = ("0", "false", "no", "off")


@dataclass(frozen=True)
class PluginSettings:
    plan: str
    attention: str = "kernels"
    balance: bool = True
    decompress_tokens: int = 0
    profile: str = ""

    def __post_init__(self):
        if not self.plan:
            raise ValueError(
                "the FastDC backend needs a compression plan: pass --additional-config "
                "'{\"fastdc\": {\"plan\": \"<plan>.safetensors\"}}' or set FASTDC_PLAN"
            )
        if self.attention not in ATTENTION_CHOICES:
            raise ValueError(f"attention must be one of {ATTENTION_CHOICES}, got {self.attention!r}")
        if self.decompress_tokens < 0:
            raise ValueError("decompress_tokens must be >= 0")


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in _TRUE:
        return True
    if text in _FALSE:
        return False
    raise ValueError(f"expected a boolean, got {value!r}")


_CONVERTERS = {"plan": str, "attention": lambda v: str(v).strip().lower(), "balance": _to_bool,
               "decompress_tokens": int, "profile": str}


def settings_from(options: dict[str, Any] | None) -> PluginSettings:
    """Settings from the "fastdc" entry of the additional config, completed from the environment."""
    options = dict(options or {})
    unknown = sorted(set(options) - set(_CONVERTERS))
    if unknown:
        raise ValueError(f"unknown FastDC options {unknown}; known options: {sorted(_CONVERTERS)}")
    values: dict[str, Any] = {}
    for name, convert in _CONVERTERS.items():
        raw = options.get(name, os.environ.get(ENV_PREFIX + name.upper()))
        if raw is not None and raw != "":
            values[name] = convert(raw)
    values.setdefault("plan", "")
    return PluginSettings(**values)


def settings_for(vllm_config) -> PluginSettings:
    """Settings of the engine described by `vllm_config`."""
    additional = getattr(vllm_config, "additional_config", None) or {}
    options = additional.get(ADDITIONAL_CONFIG_KEY) if isinstance(additional, dict) else None
    if options is not None and not isinstance(options, dict):
        raise ValueError(f"additional_config[{ADDITIONAL_CONFIG_KEY!r}] must be a JSON object")
    return settings_from(options)
