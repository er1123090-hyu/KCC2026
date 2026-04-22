"""Runtime compatibility shims for local vLLM server launches."""

try:
    import huggingface_hub
    from huggingface_hub import constants as _hf_constants

    if not hasattr(huggingface_hub, "is_offline_mode"):
        def is_offline_mode() -> bool:
            return bool(getattr(_hf_constants, "HF_HUB_OFFLINE", False))

        huggingface_hub.is_offline_mode = is_offline_mode
except Exception:
    pass
