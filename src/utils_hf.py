"""Hugging Face and local model helpers."""

from __future__ import annotations

import gc
import os
from typing import Any

import numpy as np

_TEXT_GEN_CACHE: dict[str, tuple[Any, Any]] = {}
_EMBEDDER_CACHE: dict[str, Any] = {}


def _ensure_gpu_visibility() -> None:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1")


def unload_all_models() -> None:
    _TEXT_GEN_CACHE.clear()
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _load_text_generation_model(model_id: str):
    _ensure_gpu_visibility()
    if model_id in _TEXT_GEN_CACHE:
        return _TEXT_GEN_CACHE[model_id]
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:
        raise RuntimeError(f"Failed to import HF generation dependencies for {model_id}: {exc}") from exc
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype="auto",
            device_map="auto",
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to load rubric generator model: {model_id}") from exc
    _TEXT_GEN_CACHE[model_id] = (tokenizer, model)
    return tokenizer, model


def generate_text_with_hf(
    *,
    model_id: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float | None = None,
) -> str:
    tokenizer, model = _load_text_generation_model(model_id)
    try:
        import torch
    except Exception as exc:
        raise RuntimeError(f"Failed to import torch for {model_id}: {exc}") from exc

    messages = [{"role": "user", "content": prompt}]
    try:
        rendered_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        rendered_prompt = prompt

    inputs = tokenizer(rendered_prompt, return_tensors="pt")
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = getattr(model, "device", "cpu")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "temperature": temperature if temperature > 0 else 1.0,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if top_p is not None:
        generation_kwargs["top_p"] = top_p

    with torch.no_grad():
        outputs = model.generate(**inputs, **generation_kwargs)
    generated = outputs[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def _load_embedder(model_name: str):
    _ensure_gpu_visibility()
    if model_name in _EMBEDDER_CACHE:
        return _EMBEDDER_CACHE[model_name]
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:
        raise RuntimeError(f"Failed to import sentence-transformers for {model_name}: {exc}") from exc
    try:
        device = os.environ.get("KCC_EMBED_DEVICE", "cpu")
        if device == "auto":
            device = "cuda" if _cuda_available() else "cpu"
        embedder = SentenceTransformer(model_name, device=device)
    except Exception as exc:
        raise RuntimeError(f"Failed to load embedding model: {model_name}") from exc
    _EMBEDDER_CACHE[model_name] = embedder
    return embedder


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def embed_texts(texts: list[str], *, model_name: str) -> np.ndarray:
    embedder = _load_embedder(model_name)
    vectors = embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(vectors, dtype=float)
