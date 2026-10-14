"""vLLM integration of FastDC (registered through the `vllm.general_plugins` entry point).

    vllm serve <model> --attention-backend CUSTOM --additional-config '{"fastdc": {"plan": "<plan>.safetensors"}}'

The plan file (fastdc.plan.CompressionPlan) selects the method: full, fastdc or palu. See `settings.py` for
the options and `backend.py` for the backend.
"""

BACKEND_CLASS = "fastdc.vllm_plugin.backend.FastDCAttentionBackend"


def register() -> None:
    """Entry point of the `vllm.general_plugins` group: registers the backend as CUSTOM and the load hook."""
    from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend

    from fastdc.vllm_plugin.loader import install_load_hook

    register_backend(AttentionBackendEnum.CUSTOM, BACKEND_CLASS)
    install_load_hook()
