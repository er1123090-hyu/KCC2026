"""OpenAI-compatible API helpers for the Korean rubric grounding experiment."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Iterable

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from .utils import ensure_parent, extract_first_json_object


def _get_sdk():
    from openai import OpenAI

    return OpenAI


def get_openai_api_key(config: dict[str, Any], *, allow_missing: bool = False) -> str | None:
    env_name = config["api"]["openai_api_key_env"]
    api_key = os.environ.get(env_name)
    if api_key:
        return api_key
    if allow_missing:
        return None
    raise RuntimeError(f"Missing required OpenAI API key env var: {env_name}")


def get_openai_client(config: dict[str, Any]):
    OpenAI = _get_sdk()
    api_key = get_openai_api_key(config)
    base_url = os.environ.get("OPENAI_BASE_URL") or config["api"].get("openai_base_url")
    kwargs: dict[str, Any] = {
        "api_key": api_key,
        "timeout": config["api"]["request_timeout_seconds"],
        "max_retries": 0,
    }
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def _message_to_text(message: Any) -> str:
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                chunks.append(str(item.get("text", "")))
            else:
                text = getattr(item, "text", None)
                if text:
                    chunks.append(str(text))
        return "\n".join(chunk for chunk in chunks if chunk)
    return str(content or "")


def _candidate_model_ids(model: str) -> list[str]:
    candidates = [model]
    if "/" not in model and model.startswith("gpt-oss"):
        candidates.append(f"openai/{model}")
    return candidates


def _should_disable_qwen_thinking(model: str) -> bool:
    raw_value = os.environ.get("KCC_DISABLE_QWEN_THINKING", "1").strip().lower()
    disable_requested = raw_value not in {"0", "false", "no"}
    return disable_requested and "qwen3" in model.lower()


def _merge_extra_body_for_model(model: str, extra_body: dict[str, Any] | None) -> dict[str, Any] | None:
    merged = dict(extra_body or {})
    if _should_disable_qwen_thinking(model):
        chat_template_kwargs = dict(merged.get("chat_template_kwargs") or {})
        chat_template_kwargs["enable_thinking"] = False
        merged["chat_template_kwargs"] = chat_template_kwargs
    return merged or None


def _responses_output_to_text(output: Any) -> str:
    chunks: list[str] = []
    for item in output or []:
        item_type = item.get("type") if isinstance(item, dict) else getattr(item, "type", None)
        if item_type != "message":
            continue
        content = item.get("content") if isinstance(item, dict) else getattr(item, "content", None)
        for block in content or []:
            block_type = block.get("type") if isinstance(block, dict) else getattr(block, "type", None)
            if block_type in {"output_text", "text"}:
                text = block.get("text") if isinstance(block, dict) else getattr(block, "text", None)
                if text:
                    chunks.append(str(text))
    return "\n".join(chunk for chunk in chunks if chunk).strip()


@retry(
    stop=stop_after_attempt(4),
    wait=wait_fixed(2),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def create_chat_completion(
    *,
    config: dict[str, Any],
    model: str,
    prompt: str,
    temperature: float | None,
    max_output_tokens: int,
    top_p: float | None = None,
    response_format_json: bool = False,
    reasoning_effort: str | None = None,
    extra_body: dict[str, Any] | None = None,
) -> str:
    client = get_openai_client(config)
    last_exception: Exception | None = None
    for candidate_model in _candidate_model_ids(model):
        merged_extra_body = _merge_extra_body_for_model(candidate_model, extra_body)
        body: dict[str, Any] = {
            "model": candidate_model,
            "messages": [{"role": "user", "content": prompt}],
        }
        if temperature is not None:
            body["temperature"] = temperature
        if top_p is not None:
            body["top_p"] = top_p
        if response_format_json:
            body["response_format"] = {"type": "json_object"}
        if reasoning_effort is not None:
            body["reasoning_effort"] = reasoning_effort
        if merged_extra_body:
            body["extra_body"] = merged_extra_body

        for max_key in ("max_completion_tokens", "max_tokens"):
            trial_body = dict(body)
            trial_body[max_key] = max_output_tokens
            try:
                response = client.chat.completions.create(**trial_body)
                return _message_to_text(response.choices[0].message)
            except TypeError:
                continue
            except Exception as exc:
                last_exception = exc
                message = str(exc)
                unsupported_this_key = (
                    max_key in message
                    or (max_key == "max_completion_tokens" and "max_completion_tokens" in message)
                    or (max_key == "max_tokens" and "max_tokens" in message)
                )
                if unsupported_this_key and max_key == "max_completion_tokens":
                    continue
                model_not_found = "model_not_found" in message or "does not exist" in message
                if model_not_found and candidate_model != _candidate_model_ids(model)[-1]:
                    break
                raise exc
    if last_exception is not None:
        raise last_exception
    raise RuntimeError("Failed to create chat completion with supported max token parameters.")


def parse_json_response(raw_text: str) -> dict[str, Any]:
    return extract_first_json_object(raw_text)


def _augment_openai_error(
    exc: Exception,
    *,
    model: str,
    temperature: float | None,
    top_p: float | None,
) -> RuntimeError:
    message = str(exc)
    detail = f"OpenAI-compatible request failed for model={model}."
    if "temperature" in message and "unsupported" in message.lower():
        detail += (
            f" Backend rejected temperature={temperature}"
            + (f" and top_p={top_p}" if top_p is not None else "")
            + "."
        )
        if model.startswith("gpt-5"):
            detail += " On the current OpenAI backend, GPT-5 rejects non-default sampling controls."
    elif "model_not_found" in message or "does not exist or you do not have access" in message:
        detail += (
            " The configured backend does not expose this model or the current API key lacks access."
            " If you intend to use a local/OpenAI-compatible server, set OPENAI_BASE_URL accordingly."
        )
    return RuntimeError(f"{detail} Original error: {message}")


def preflight_chat_completion(
    *,
    config: dict[str, Any],
    model: str,
    prompt: str,
    temperature: float | None,
    max_output_tokens: int,
    top_p: float | None = None,
    reasoning_effort: str | None = None,
) -> None:
    try:
        create_chat_completion(
            config=config,
            model=model,
            prompt=prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            top_p=top_p,
            response_format_json=False,
            reasoning_effort=reasoning_effort,
        )
    except Exception as exc:
        raise _augment_openai_error(exc, model=model, temperature=temperature, top_p=top_p) from exc


@retry(
    stop=stop_after_attempt(4),
    wait=wait_fixed(2),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def create_response_text(
    *,
    config: dict[str, Any],
    model: str,
    prompt: str,
    max_output_tokens: int,
    reasoning_effort: str | None = None,
) -> str:
    client = get_openai_client(config)
    body: dict[str, Any] = {
        "model": model,
        "input": prompt,
        "max_output_tokens": max_output_tokens,
    }
    if reasoning_effort is not None:
        body["reasoning"] = {"effort": reasoning_effort}
    response = client.responses.create(**body)
    text = getattr(response, "output_text", None) or _responses_output_to_text(getattr(response, "output", None))
    return str(text or "").strip()


def preflight_response(
    *,
    config: dict[str, Any],
    model: str,
    prompt: str,
    max_output_tokens: int,
    reasoning_effort: str | None = None,
) -> None:
    text = create_response_text(
        config=config,
        model=model,
        prompt=prompt,
        max_output_tokens=max_output_tokens,
        reasoning_effort=reasoning_effort,
    )
    if not text:
        raise RuntimeError(f"Responses API preflight for model={model} returned empty visible text.")


def _serialize_batch_lines(requests: Iterable[dict[str, Any]]) -> bytes:
    return "\n".join(json.dumps(row, ensure_ascii=False) for row in requests).encode("utf-8")


def _response_content_to_text(content_response: Any) -> str:
    if hasattr(content_response, "text"):
        return str(content_response.text)
    if hasattr(content_response, "read"):
        payload = content_response.read()
        if isinstance(payload, bytes):
            return payload.decode("utf-8")
        return str(payload)
    if isinstance(content_response, (bytes, bytearray)):
        return bytes(content_response).decode("utf-8")
    return str(content_response)


def run_chat_completion_batch(
    *,
    config: dict[str, Any],
    requests: list[dict[str, Any]],
    batch_filename: str,
    metadata: dict[str, str] | None = None,
    poll_seconds: int = 10,
) -> list[dict[str, Any]]:
    if not requests:
        return []
    client = get_openai_client(config)
    archive_path = ensure_parent(Path("batch_requests") / batch_filename)
    archive_path.write_bytes(_serialize_batch_lines(requests))
    file_obj = client.files.create(
        file=(batch_filename, _serialize_batch_lines(requests)),
        purpose="batch",
    )
    batch = client.batches.create(
        input_file_id=file_obj.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata=metadata or {},
    )

    terminal_statuses = {"completed", "failed", "cancelled", "expired"}
    while batch.status not in terminal_statuses:
        time.sleep(poll_seconds)
        batch = client.batches.retrieve(batch.id)

    if batch.status != "completed":
        raise RuntimeError(f"Batch {batch.id} ended with status={batch.status}")
    error_text = ""
    if batch.error_file_id:
        error_text = _response_content_to_text(client.files.content(batch.error_file_id))
        error_path = ensure_parent(Path("batch_errors") / f"{batch.id}.jsonl")
        error_path.write_text(error_text, encoding="utf-8")
    if not batch.output_file_id:
        detail = f"Batch {batch.id} completed without an output file."
        if error_text:
            detail += f" First batch error: {error_text.splitlines()[0][:1000]}"
        raise RuntimeError(detail)

    content = client.files.content(batch.output_file_id)
    output_text = _response_content_to_text(content)
    results = [json.loads(line) for line in output_text.splitlines() if line.strip()]
    if error_text:
        error_rows = [json.loads(line) for line in error_text.splitlines() if line.strip()]
        if error_rows:
            first_error = error_rows[0]
            response_body = first_error.get("response", {}).get("body", {})
            raise RuntimeError(
                f"Batch {batch.id} completed with row-level errors. "
                f"First error for custom_id={first_error.get('custom_id')}: {response_body}"
            )
    return results


def run_responses_batch(
    *,
    config: dict[str, Any],
    requests: list[dict[str, Any]],
    batch_filename: str,
    metadata: dict[str, str] | None = None,
    poll_seconds: int = 10,
) -> list[dict[str, Any]]:
    if not requests:
        return []
    client = get_openai_client(config)
    archive_path = ensure_parent(Path("batch_requests") / batch_filename)
    archive_path.write_bytes(_serialize_batch_lines(requests))
    file_obj = client.files.create(
        file=(batch_filename, _serialize_batch_lines(requests)),
        purpose="batch",
    )
    batch = client.batches.create(
        input_file_id=file_obj.id,
        endpoint="/v1/responses",
        completion_window="24h",
        metadata=metadata or {},
    )

    terminal_statuses = {"completed", "failed", "cancelled", "expired"}
    while batch.status not in terminal_statuses:
        time.sleep(poll_seconds)
        batch = client.batches.retrieve(batch.id)

    if batch.status != "completed":
        raise RuntimeError(f"Batch {batch.id} ended with status={batch.status}")
    error_text = ""
    if batch.error_file_id:
        error_text = _response_content_to_text(client.files.content(batch.error_file_id))
        error_path = ensure_parent(Path("batch_errors") / f"{batch.id}.jsonl")
        error_path.write_text(error_text, encoding="utf-8")
    if not batch.output_file_id:
        detail = f"Batch {batch.id} completed without an output file."
        if error_text:
            detail += f" First batch error: {error_text.splitlines()[0][:1000]}"
        raise RuntimeError(detail)

    output_text = _response_content_to_text(client.files.content(batch.output_file_id))
    results = [json.loads(line) for line in output_text.splitlines() if line.strip()]
    if error_text:
        error_rows = [json.loads(line) for line in error_text.splitlines() if line.strip()]
        if error_rows:
            first_error = error_rows[0]
            response_body = first_error.get("response", {}).get("body", {})
            raise RuntimeError(
                f"Batch {batch.id} completed with row-level errors. "
                f"First error for custom_id={first_error.get('custom_id')}: {response_body}"
            )
    return results


def write_batch_request_archive(path: str | Path, requests: list[dict[str, Any]]) -> None:
    target = ensure_parent(path)
    target.write_bytes(_serialize_batch_lines(requests))
