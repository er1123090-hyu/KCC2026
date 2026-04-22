"""Generate or backfill exactly 8 auxiliary sampled responses per prompt."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from .schemas import AuxiliarySample
from .utils import (
    detect_auxiliary_reuse,
    get_auxiliary_samples_path,
    get_meta_eval_pairs_path,
    grouped_prompt_rows,
    load_config,
    load_meta_eval_pairs,
    load_auxiliary_samples,
    parse_common_args,
    stable_hash,
    write_jsonl,
)
from .utils_openai import preflight_response, run_responses_batch


def _mock_samples(prompt_id: str, prompt: str, config: dict[str, Any]) -> list[AuxiliarySample]:
    samples: list[AuxiliarySample] = []
    params = {
        "temperature": config["auxiliary_response_generation"]["temperature"],
        "top_p": config["auxiliary_response_generation"]["top_p"],
        "max_new_tokens": config["auxiliary_response_generation"]["max_new_tokens"],
        "reasoning_effort": config["auxiliary_response_generation"].get("reasoning_effort"),
        "smoke_mock": True,
    }
    for index in range(1, config["auxiliary_response_generation"]["num_samples"] + 1):
        fingerprint = stable_hash(prompt_id, index, prefix="smoke")
        response = (
            f"[SMOKE SAMPLE {index:02d}] {fingerprint} "
            f"이 응답은 prompt_id={prompt_id}에 대한 batch-api 대체 mock입니다. "
            f"질문 요약: {prompt[:120]}"
        )
        samples.append(
            AuxiliarySample(
                prompt_id=prompt_id,
                sample_id=f"sample_{index:02d}",
                response=response,
                generator_model=config["auxiliary_response_generation"]["model"],
                sampling_params=params,
            )
        )
    return samples


def _build_batch_requests(
    prompt_rows: dict[str, dict[str, Any]],
    missing_sample_ids: dict[str, list[str]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    requests: list[dict[str, Any]] = []
    generation_cfg = config["auxiliary_response_generation"]
    for prompt_id, sample_ids in sorted(missing_sample_ids.items()):
        prompt = prompt_rows[prompt_id]["prompt"]
        for sample_id in sample_ids:
            requests.append(
                {
                    "custom_id": f"{prompt_id}::{sample_id}",
                    "method": "POST",
                    "url": "/v1/responses",
                    "body": {
                        "model": generation_cfg["model"],
                        "input": prompt,
                        "max_output_tokens": generation_cfg["max_new_tokens"],
                    },
                }
            )
            if generation_cfg.get("reasoning_effort") is not None:
                requests[-1]["body"]["reasoning"] = {"effort": generation_cfg["reasoning_effort"]}
    return requests


def _parse_batch_results(results: list[dict[str, Any]], config: dict[str, Any]) -> list[AuxiliarySample]:
    rows: list[AuxiliarySample] = []
    for result in results:
        custom_id = result["custom_id"]
        prompt_id, sample_id = custom_id.split("::", maxsplit=1)
        body = result.get("response", {}).get("body", {})
        content = body.get("output_text") or ""
        if not content:
            for item in body.get("output") or []:
                if item.get("type") != "message":
                    continue
                for block in item.get("content") or []:
                    if block.get("type") in {"output_text", "text"} and block.get("text"):
                        content += str(block.get("text"))
        if not content.strip():
            raise RuntimeError(f"Responses batch returned empty visible content for {custom_id}")
        rows.append(
            AuxiliarySample(
                prompt_id=prompt_id,
                sample_id=sample_id,
                response=str(content).strip(),
                generator_model=config["auxiliary_response_generation"]["model"],
                sampling_params={
                    "temperature": config["auxiliary_response_generation"]["temperature"],
                    "top_p": config["auxiliary_response_generation"]["top_p"],
                    "max_new_tokens": config["auxiliary_response_generation"]["max_new_tokens"],
                    "reasoning_effort": config["auxiliary_response_generation"].get("reasoning_effort"),
                    "batch_api": True,
                },
            )
        )
    return rows


def _candidate_existing_paths() -> list[Path]:
    paths = [
        get_auxiliary_samples_path(),
        get_auxiliary_samples_path("calibration"),
        get_auxiliary_samples_path("test"),
    ]
    deduped: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    return deduped


def _load_existing_samples_by_prompt() -> dict[str, dict[str, AuxiliarySample]]:
    existing: dict[str, dict[str, AuxiliarySample]] = defaultdict(dict)
    for path in _candidate_existing_paths():
        if not path.exists():
            continue
        for sample in load_auxiliary_samples(path):
            existing[sample.prompt_id][sample.sample_id] = sample
    return existing


def main() -> None:
    args = parse_common_args()
    config = load_config(args.config, smoke_test=args.smoke_test)
    pairs = load_meta_eval_pairs(get_meta_eval_pairs_path())
    prompt_rows = grouped_prompt_rows(pairs)
    expected = config["auxiliary_response_generation"]["num_samples"]

    existing_by_prompt = _load_existing_samples_by_prompt()
    missing_sample_ids: dict[str, list[str]] = {}
    merged_rows: dict[tuple[str, str], AuxiliarySample] = {}

    for prompt_id, payload in sorted(prompt_rows.items()):
        complete_ids = {f"sample_{index:02d}" for index in range(1, expected + 1)}
        existing_samples = existing_by_prompt.get(prompt_id, {})
        missing = sorted(complete_ids - set(existing_samples))
        for sample_id, sample in existing_samples.items():
            merged_rows[(prompt_id, sample_id)] = sample
        if missing:
            missing_sample_ids[prompt_id] = missing

    if config["experiment"]["smoke_test"]:
        for prompt_id, payload in sorted(prompt_rows.items()):
            for sample in _mock_samples(prompt_id, payload["prompt"], config):
                merged_rows[(sample.prompt_id, sample.sample_id)] = sample
    elif missing_sample_ids:
        preflight_response(
            config=config,
            model=config["auxiliary_response_generation"]["model"],
            prompt="한국어로 한 문장만 답해 주세요. 준비 완료라고만 말해 주세요.",
            max_output_tokens=config["auxiliary_response_generation"]["max_new_tokens"],
            reasoning_effort=config["auxiliary_response_generation"].get("reasoning_effort"),
        )
        requests = _build_batch_requests(prompt_rows, missing_sample_ids, config)
        results = run_responses_batch(
            config=config,
            requests=requests,
            batch_filename="auxiliary_samples_full.jsonl",
            metadata={"purpose": "auxiliary_samples_full"},
        )
        for sample in _parse_batch_results(results, config):
            merged_rows[(sample.prompt_id, sample.sample_id)] = sample

    merged_list = sorted(merged_rows.values(), key=lambda item: (item.prompt_id, item.sample_id))
    samples_by_prompt: dict[str, list[AuxiliarySample]] = defaultdict(list)
    for sample in merged_list:
        samples_by_prompt[sample.prompt_id].append(sample)

    bad_counts = {
        prompt_id: len(samples)
        for prompt_id, samples in samples_by_prompt.items()
        if len(samples) != expected
    }
    if bad_counts:
        raise RuntimeError(f"Incomplete auxiliary sample generation for prompts: {list(sorted(bad_counts.items()))[:5]}")

    detect_auxiliary_reuse(pairs, samples_by_prompt)
    write_jsonl(get_auxiliary_samples_path(), [sample.model_dump(mode="json") for sample in merged_list])
    complete_prompt_count = sum(1 for prompt_id in prompt_rows if len(samples_by_prompt[prompt_id]) == expected)
    print(
        f"Wrote {len(merged_list)} auxiliary samples for {complete_prompt_count} prompts "
        f"to {get_auxiliary_samples_path()}"
    )


if __name__ == "__main__":
    main()
