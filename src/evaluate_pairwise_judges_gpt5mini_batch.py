"""Submit / inspect / collect GPT-5-mini Batch API runs for pairwise-judge evaluation."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from .evaluate_pairs import _build_task_context, _parse_pairwise_judge_output
from .schemas import PairPrediction, Rubric
from .utils import (
    build_rar_rubric_text,
    get_condition_specs,
    get_meta_eval_pairs_path,
    load_config,
    load_meta_eval_pairs,
    load_prompt_template,
    read_jsonl,
    write_json,
    write_jsonl,
)
from .utils_openai import get_openai_client


def _meta_path() -> Path:
    return Path("/data/minseo/KCC2026/results/raw/pair_predictions_pairwise_judge_gpt5mini_batch.meta.json")


def _output_path() -> Path:
    return Path("/data/minseo/KCC2026/results/raw/pair_predictions_pairwise_judge_gpt5mini_batch.jsonl")


def _requests_archive_path() -> Path:
    return Path("/data/minseo/KCC2026/batch_requests/pair_predictions_pairwise_judge_gpt5mini_batch.jsonl")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=["submit", "status", "collect"], required=True)
    parser.add_argument("--batch_id")
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--poll_seconds", type=int, default=15)
    parser.add_argument("--completion_window", default="24h")
    return parser


def _spec_key(method: str, generator_family: str | None) -> str:
    return method if generator_family is None else f"{method}__{generator_family}"


def _custom_id(method: str, generator_family: str | None, pair_id: str) -> str:
    family = generator_family or "none"
    return f"{method}::{family}::{pair_id}"


def _parse_custom_id(custom_id: str) -> tuple[str, str | None, str]:
    method, family, pair_id = custom_id.split("::", maxsplit=2)
    return method, (None if family == "none" else family), pair_id


def _load_rubric_lookup(config: dict[str, object]) -> dict[str, dict[str, Rubric]]:
    lookup: dict[str, dict[str, Rubric]] = {}
    for spec in get_condition_specs(config):
        if spec.eval_protocol != "pairwise_judge" or spec.rubric_scope == "none":
            continue
        assert spec.generator_family is not None
        path = Path(f"/data/minseo/KCC2026/data/processed/rubrics_{spec.method}__{spec.generator_family}.jsonl")
        rows = [Rubric.model_validate(row) for row in read_jsonl(path)]
        if spec.rubric_scope == "pair":
            lookup[spec.condition_name] = {str(rubric.pair_id): rubric for rubric in rows if rubric.pair_id}
        else:
            lookup[spec.condition_name] = {rubric.prompt_id: rubric for rubric in rows}
    return lookup


def _build_requests(config: dict[str, Any], *, model: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pairs = load_meta_eval_pairs(get_meta_eval_pairs_path())
    specs = [spec for spec in get_condition_specs(config) if spec.eval_protocol == "pairwise_judge"]
    rubric_lookup = _load_rubric_lookup(config)
    baseline_template = load_prompt_template("configs/prompts/pairwise_baseline_judge_ko.txt")
    rubric_judge_template = load_prompt_template("configs/prompts/pairwise_rubric_judge_ko.txt")

    requests: list[dict[str, Any]] = []
    request_index: dict[str, Any] = {}
    for spec in specs:
        for pair in pairs:
            task_context = _build_task_context(pair.prompt)
            if spec.method == "pairwise_baseline":
                prompt_text = baseline_template.format(
                    task_context=task_context,
                    response_a=pair.response_a,
                    response_b=pair.response_b,
                )
            else:
                rubric = rubric_lookup[spec.condition_name][pair.pair_id if spec.rubric_scope == "pair" else pair.prompt_id]
                prompt_text = rubric_judge_template.format(
                    task_context=task_context,
                    rubric_text=build_rar_rubric_text(rubric.rar_items),
                    response_a=pair.response_a,
                    response_b=pair.response_b,
                )
            custom_id = _custom_id(spec.method, spec.generator_family, pair.pair_id)
            body: dict[str, Any] = {
                "model": model,
                "messages": [{"role": "user", "content": prompt_text}],
                "temperature": config["pairwise_evaluation"]["temperature"],
                "response_format": {"type": "json_object"},
                "max_completion_tokens": config["pairwise_evaluation"]["max_output_tokens"],
            }
            reasoning_effort = config["pairwise_evaluation"].get("reasoning_effort")
            if reasoning_effort is not None:
                body["reasoning_effort"] = reasoning_effort
            requests.append(
                {
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": body,
                }
            )
            request_index[custom_id] = {
                "pair_id": pair.pair_id,
                "gold_preference": pair.gold_preference,
                "method": spec.method,
                "generator_family": spec.generator_family,
                "generator_model": spec.generator_model,
            }
    return requests, request_index


def _submit_batch(
    *,
    config: dict[str, Any],
    requests: list[dict[str, Any]],
    request_index: dict[str, Any],
    model: str,
    completion_window: str,
) -> dict[str, Any]:
    client = get_openai_client(config)
    archive_path = _requests_archive_path()
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    payload_bytes = "\n".join(json.dumps(row, ensure_ascii=False) for row in requests).encode("utf-8")
    archive_path.write_bytes(payload_bytes)
    file_obj = client.files.create(
        file=(archive_path.name, payload_bytes),
        purpose="batch",
    )
    batch = client.batches.create(
        input_file_id=file_obj.id,
        endpoint="/v1/chat/completions",
        completion_window=completion_window,
        metadata={
            "purpose": "pairwise_judges_gpt5mini",
            "model": model,
        },
    )
    meta = {
        "batch_id": batch.id,
        "status": batch.status,
        "model": model,
        "request_count": len(requests),
        "input_file_id": file_obj.id,
        "archive_path": str(archive_path),
        "output_path": str(_output_path()),
        "request_index_path": str(archive_path.with_suffix(".index.json")),
        "completion_window": completion_window,
        "submitted_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    write_json(_meta_path(), meta)
    write_json(archive_path.with_suffix(".index.json"), request_index)
    return meta


def _load_meta(batch_id: str | None) -> dict[str, Any]:
    meta = json.loads(_meta_path().read_text(encoding="utf-8"))
    if batch_id and meta["batch_id"] != batch_id:
        raise RuntimeError(f"Requested batch_id={batch_id} does not match saved meta batch_id={meta['batch_id']}")
    return meta


def _poll_batch(config: dict[str, Any], batch_id: str, poll_seconds: int) -> Any:
    client = get_openai_client(config)
    terminal_statuses = {"completed", "failed", "cancelled", "expired"}
    batch = client.batches.retrieve(batch_id)
    while batch.status not in terminal_statuses:
        time.sleep(poll_seconds)
        batch = client.batches.retrieve(batch_id)
    return batch


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


def _collect_batch(config: dict[str, Any], *, batch_id: str, poll_seconds: int) -> dict[str, Any]:
    meta = _load_meta(batch_id)
    client = get_openai_client(config)
    batch = _poll_batch(config, batch_id, poll_seconds)
    if batch.status != "completed":
        raise RuntimeError(f"Batch {batch.id} ended with status={batch.status}")

    error_rows: list[dict[str, Any]] = []
    if batch.error_file_id:
        error_text = _response_content_to_text(client.files.content(batch.error_file_id))
        error_path = Path("/data/minseo/KCC2026/batch_errors") / f"{batch.id}.jsonl"
        error_path.parent.mkdir(parents=True, exist_ok=True)
        error_path.write_text(error_text, encoding="utf-8")
        error_rows = [json.loads(line) for line in error_text.splitlines() if line.strip()]

    if not batch.output_file_id:
        raise RuntimeError(f"Batch {batch.id} completed without an output file.")

    output_text = _response_content_to_text(client.files.content(batch.output_file_id))
    result_rows = [json.loads(line) for line in output_text.splitlines() if line.strip()]
    request_index = json.loads(Path(meta["request_index_path"]).read_text(encoding="utf-8"))

    parsed_rows: list[dict[str, Any]] = []
    for result in result_rows:
        custom_id = result["custom_id"]
        descriptor = request_index[custom_id]
        response_body = result.get("response", {}).get("body", {})
        status_code = result.get("response", {}).get("status_code")
        raw_output = ""
        winner = "A"
        justification = ""
        parse_failure = False
        if status_code == 200 and response_body.get("choices"):
            message = response_body["choices"][0]["message"]
            raw_output = str(message.get("content") or "")
            try:
                winner, justification = _parse_pairwise_judge_output(raw_output)
            except Exception as exc:
                justification = f"pairwise_parse_failure:{exc}"
                parse_failure = True
        else:
            parse_failure = True
            raw_output = json.dumps(response_body, ensure_ascii=False)
            justification = f"batch_row_error:status_{status_code}"
        parsed_rows.append(
            PairPrediction(
                pair_id=descriptor["pair_id"],
                method=descriptor["method"],
                generator_family=descriptor["generator_family"],
                generator_model=descriptor["generator_model"],
                eval_protocol="pairwise_judge",
                pred_preference=winner,
                gold_preference=descriptor["gold_preference"],
                justification=justification,
                parse_failure=parse_failure,
                raw_output=raw_output,
            ).model_dump(mode="json")
        )

    for error_row in error_rows:
        custom_id = error_row.get("custom_id")
        if not custom_id or custom_id not in request_index:
            continue
        descriptor = request_index[custom_id]
        response_body = error_row.get("response", {}).get("body", {})
        parsed_rows.append(
            PairPrediction(
                pair_id=descriptor["pair_id"],
                method=descriptor["method"],
                generator_family=descriptor["generator_family"],
                generator_model=descriptor["generator_model"],
                eval_protocol="pairwise_judge",
                pred_preference="A",
                gold_preference=descriptor["gold_preference"],
                justification=f"batch_row_error:{response_body}",
                parse_failure=True,
                raw_output=json.dumps(response_body, ensure_ascii=False),
            ).model_dump(mode="json")
        )

    parsed_rows.sort(key=lambda row: (row["method"], str(row.get("generator_family") or ""), row["pair_id"]))
    write_jsonl(_output_path(), parsed_rows)

    summary = {
        "batch_id": batch.id,
        "status": batch.status,
        "output_rows": len(parsed_rows),
        "parse_failures": sum(1 for row in parsed_rows if row.get("parse_failure")),
        "output_path": str(_output_path()),
        "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    write_json(_meta_path(), {**meta, **summary})
    return summary


def main() -> None:
    args = _build_arg_parser().parse_args()
    config = load_config(args.config)

    if args.mode == "submit":
        requests, request_index = _build_requests(config, model=args.model)
        meta = _submit_batch(
            config=config,
            requests=requests,
            request_index=request_index,
            model=args.model,
            completion_window=args.completion_window,
        )
        print(json.dumps(meta, ensure_ascii=False, indent=2))
        return

    if args.mode == "status":
        meta = _load_meta(args.batch_id)
        client = get_openai_client(config)
        batch = client.batches.retrieve(meta["batch_id"])
        print(json.dumps({"batch_id": batch.id, "status": batch.status}, ensure_ascii=False, indent=2))
        return

    if args.mode == "collect":
        meta = _load_meta(args.batch_id)
        summary = _collect_batch(config, batch_id=meta["batch_id"], poll_seconds=args.poll_seconds)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    raise RuntimeError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()
