"""Generate pairwise RaR / RRD rubrics for the Korean experiment."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .schemas import AuxiliarySample, Criterion, MetaEvalPair, RaRRubricItem, Rubric
from .utils import (
    append_jsonl,
    get_auxiliary_samples_path,
    get_meta_eval_pairs_path,
    get_rubric_output_path,
    grouped_prompt_rows,
    load_auxiliary_samples,
    load_config,
    load_meta_eval_pairs,
    load_prompt_template,
    read_json,
    read_jsonl,
    render_prompt_template,
    stable_hash,
    write_json,
    write_jsonl,
)
from .utils import extract_first_json_object
from .utils_hf import embed_texts, generate_text_with_hf, unload_all_models
from .utils_openai import create_chat_completion, parse_json_response

_AXIS_TEMPLATES = [
    ("instruction_following", "essential", 1.0, "positive", "응답은 사용자 요청을 직접적이고 빠짐없이 수행한다."),
    ("correctness", "essential", 1.0, "positive", "응답은 사실 오류와 근거 없는 단정을 피하고 정확한 내용을 제공한다."),
    ("completeness", "important", 0.7, "positive", "응답은 핵심 요소를 누락하지 않고 필요한 설명을 포함한다."),
    ("korean_naturalness", "important", 0.7, "positive", "응답은 자연스럽고 읽기 쉬운 한국어 문장을 사용한다."),
    ("register", "important", 0.7, "positive", "응답은 상황에 맞는 존댓말과 문체를 유지한다."),
    ("conciseness", "optional", 0.3, "positive", "응답은 불필요한 반복 없이 필요한 정보만 간결하게 제시한다."),
]

_RAR_TEMPLATE_ITEMS = [
    ("핵심 충족", "Essential Criteria: 응답은 사용자의 핵심 요청을 직접적으로 수행한다.", 5),
    ("정확성", "Essential Criteria: 응답은 사실 오류와 근거 없는 단정을 피한다.", 5),
    ("완결성", "Important Criteria: 응답은 핵심 요소를 빠뜨리지 않고 필요한 설명을 포함한다.", 4),
    ("자연스러움", "Important Criteria: 응답은 자연스럽고 매끄러운 한국어 표현을 사용한다.", 3),
    ("문체 적합성", "Important Criteria: 응답은 상황에 맞는 톤과 문체를 유지한다.", 3),
    ("간결성", "Optional Criteria: 응답은 불필요한 장황함 없이 핵심을 전달한다.", 2),
    ("주의점", "Pitfall Criteria: 질문과 무관한 내용을 길게 늘어놓지 않는다.", -1),
]


def _load_output_rows(output_path: Path, *, key_field: str) -> dict[str, dict[str, Any]]:
    if not output_path.exists():
        return {}
    existing: dict[str, dict[str, Any]] = {}
    for item in read_jsonl(output_path):
        if not isinstance(item, dict):
            continue
        key_value = item.get(key_field)
        if not key_value:
            continue
        existing[str(key_value)] = item
    return existing


def _progress_path(output_path: Path) -> Path:
    return output_path.with_suffix(output_path.suffix + ".progress.json")


def _write_progress(
    *,
    output_path: Path,
    generator_family: str,
    method: str,
    done: int,
    total: int,
    key_field: str,
) -> None:
    write_json(
        _progress_path(output_path),
        {
            "generator_family": generator_family,
            "method": method,
            "output_path": str(output_path),
            "key_field": key_field,
            "completed_rows": done,
            "expected_rows": total,
            "remaining_rows": max(total - done, 0),
            "complete": done >= total,
        },
    )


def _extract_json_payload(raw_text: str) -> Any:
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.removeprefix("```json").removeprefix("```").strip()
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()
    try:
        return json.loads(cleaned)
    except Exception:
        pass
    try:
        return extract_first_json_object(cleaned)
    except Exception:
        pass
    first_bracket = cleaned.find("[")
    last_bracket = cleaned.rfind("]")
    if first_bracket != -1 and last_bracket != -1 and first_bracket < last_bracket:
        return json.loads(cleaned[first_bracket : last_bracket + 1])
    raise ValueError("Could not locate a valid JSON payload in model output.")


def _call_hf_json(
    *,
    model_id: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    retries: int,
) -> tuple[dict[str, Any], str]:
    last_exception: Exception | None = None
    last_raw = ""
    for _ in range(retries + 1):
        raw_text = generate_text_with_hf(
            model_id=model_id,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.95,
        )
        last_raw = raw_text
        try:
            payload = _extract_json_payload(raw_text)
            if isinstance(payload, list):
                return {"rubrics": payload}, raw_text
            if isinstance(payload, dict):
                return payload, raw_text
            raise ValueError("JSON payload must be an object or list.")
        except Exception as exc:
            last_exception = exc
    raise RuntimeError(f"Failed to parse JSON from {model_id}: {last_exception}; raw={last_raw[:400]!r}")


def _rubric_generation_backend() -> str:
    backend = os.environ.get("KCC_RUBRIC_GENERATION_BACKEND", "hf").strip().lower()
    if backend not in {"hf", "openai_compatible"}:
        raise RuntimeError(
            "KCC_RUBRIC_GENERATION_BACKEND must be one of: 'hf', 'openai_compatible'. "
            f"Received: {backend!r}"
        )
    return backend


def _call_model_json(
    *,
    config: dict[str, Any],
    model_id: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float | None,
    retries: int,
    response_format_json: bool = False,
) -> tuple[dict[str, Any], str]:
    if _rubric_generation_backend() == "openai_compatible":
        extra_body = None
        if "qwen3" in model_id.lower():
            # Qwen3 often emits long <think> traces before the JSON payload.
            # Disable thinking explicitly so rubric generation stays parseable.
            extra_body = {"chat_template_kwargs": {"enable_thinking": False}}
        last_exception: Exception | None = None
        last_raw = ""
        for _ in range(retries + 1):
            raw_text = create_chat_completion(
                config=config,
                model=model_id,
                prompt=prompt,
                temperature=temperature,
                max_output_tokens=max_new_tokens,
                response_format_json=response_format_json,
                extra_body=extra_body,
            )
            last_raw = raw_text
            try:
                payload = _extract_json_payload(raw_text)
                if isinstance(payload, list):
                    return {"rubrics": payload}, raw_text
                if isinstance(payload, dict):
                    return payload, raw_text
                raise ValueError("JSON payload must be an object or list.")
            except Exception as exc:
                last_exception = exc
        raise RuntimeError(f"Failed to parse JSON from {model_id}: {last_exception}; raw={last_raw[:400]!r}")
    return _call_hf_json(
        model_id=model_id,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=float(temperature or 0.0),
        retries=retries,
    )


def _call_openai_json(
    *,
    config: dict[str, Any],
    model: str,
    prompt: str,
    temperature: float | None,
    max_output_tokens: int,
    retries: int,
) -> tuple[dict[str, Any], str]:
    extra_body = None
    if "qwen3" in model.lower():
        extra_body = {"chat_template_kwargs": {"enable_thinking": False}}
    last_exception: Exception | None = None
    last_raw = ""
    for _ in range(retries + 1):
        raw_text = create_chat_completion(
            config=config,
            model=model,
            prompt=prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            response_format_json=True,
            extra_body=extra_body,
        )
        last_raw = raw_text
        try:
            return parse_json_response(raw_text), raw_text
        except Exception as exc:
            last_exception = exc
    raise RuntimeError(f"Failed to parse JSON from {model}: {last_exception}; raw={last_raw[:400]!r}")


def _criteria_from_payload(
    payload: dict[str, Any],
    *,
    key: str,
    prefix: str,
    exact_count: int | None = None,
) -> list[Criterion]:
    raw_items = payload.get(key)
    if not isinstance(raw_items, list) or not raw_items:
        raise ValueError(f"Model payload does not contain a non-empty '{key}' list.")
    criteria: list[Criterion] = []
    for index, raw_item in enumerate(raw_items, start=1):
        item = dict(raw_item)
        importance = str(item.get("importance", "")).strip().lower()
        if importance in {"high", "critical"}:
            item["importance"] = "essential"
        elif importance in {"medium", "moderate"}:
            item["importance"] = "important"
        elif importance in {"low", "minor"}:
            item["importance"] = "optional"
        polarity = str(item.get("polarity", "")).strip().lower()
        if polarity in {"plus", "beneficial"}:
            item["polarity"] = "positive"
        elif polarity in {"minus", "harmful"}:
            item["polarity"] = "negative"
        if isinstance(item.get("self_contained"), str):
            item["self_contained"] = str(item["self_contained"]).strip().lower() in {"true", "yes", "1"}
        item.setdefault("id", f"{prefix}{index}")
        item.setdefault("axis", "instruction_following")
        item.setdefault("importance", "important")
        item.setdefault("weight", 0.5)
        item.setdefault("polarity", "positive")
        item.setdefault("self_contained", True)
        criteria.append(Criterion.model_validate(item))
    if exact_count is not None:
        if len(criteria) > exact_count:
            criteria = criteria[:exact_count]
        elif len(criteria) == 0:
            raise ValueError(f"Expected {exact_count} criteria, found 0")
    return criteria


def _parse_rar_items(payload: dict[str, Any], *, min_items: int, max_items: int) -> list[RaRRubricItem]:
    raw_items = payload.get("rubrics", payload)
    if not isinstance(raw_items, list):
        raise ValueError("RAR payload must be a list or an object with key 'rubrics'.")
    items: list[RaRRubricItem] = []
    for raw_item in raw_items:
        if not isinstance(raw_item, dict):
            raise ValueError("RAR rubric item must be an object.")
        item = dict(raw_item)
        items.append(RaRRubricItem.model_validate(item))
    if not (min_items <= len(items) <= max_items):
        raise ValueError(f"Expected {min_items}-{max_items} RaR items, found {len(items)}")
    return items


def _mock_criteria(*, prompt_id: str, method: str, count: int, prefix: str) -> list[Criterion]:
    criteria: list[Criterion] = []
    for index in range(count):
        axis, importance, weight, polarity, text = _AXIS_TEMPLATES[index % len(_AXIS_TEMPLATES)]
        criteria.append(
            Criterion(
                id=f"{prefix}{index + 1}",
                axis=axis,
                importance=importance,
                weight=weight,
                polarity=polarity,
                text_ko=f"{text} [smoke:{stable_hash(prompt_id, method, index, length=6)}]",
                self_contained=True,
            )
        )
    return criteria


def _mock_rar_items(*, prompt_id: str, method: str, count: int = 7) -> list[RaRRubricItem]:
    items: list[RaRRubricItem] = []
    for index in range(count):
        title, description, weight = _RAR_TEMPLATE_ITEMS[index % len(_RAR_TEMPLATE_ITEMS)]
        items.append(
            RaRRubricItem(
                title=f"{title}-{index + 1}",
                description=f"{description} [smoke:{stable_hash(prompt_id, method, index, length=6)}]",
                weight=weight,
            )
        )
    return items


def _sample_responses_json(samples: Sequence[AuxiliarySample]) -> str:
    return json.dumps([sample.response for sample in samples], ensure_ascii=False, indent=2)


def _select_rrd_samples(samples: Sequence[AuxiliarySample], *, config: dict[str, Any]) -> list[AuxiliarySample]:
    configured = config["rubric_generation"]["rrd_pairwise"].get("sample_response_count")
    if configured is None:
        return list(samples)
    sample_count = int(configured)
    if sample_count < 1:
        raise RuntimeError("rubric_generation.rrd_pairwise.sample_response_count must be >= 1")
    if len(samples) < sample_count:
        raise RuntimeError(f"Expected at least {sample_count} auxiliary samples, found {len(samples)}")
    return list(samples[:sample_count])


def _reference_response(pair: MetaEvalPair) -> str:
    return pair.response_a if pair.gold_preference == "A" else pair.response_b


def _prune_redundant_criteria(criteria: list[Criterion], *, config: dict[str, Any]) -> list[Criterion]:
    if len(criteria) <= 1:
        return criteria
    if config["experiment"]["smoke_test"]:
        deduped: list[Criterion] = []
        seen: set[str] = set()
        for criterion in criteria:
            if criterion.text_ko in seen:
                continue
            seen.add(criterion.text_ko)
            deduped.append(criterion)
        return deduped
    try:
        vectors = embed_texts(
            [criterion.text_ko for criterion in criteria],
            model_name=config["retrieval"]["embedding_model"],
        )
    except Exception:
        deduped = []
        seen = set()
        for criterion in criteria:
            if criterion.text_ko in seen:
                continue
            seen.add(criterion.text_ko)
            deduped.append(criterion)
        return deduped
    kept_indices: list[int] = []
    threshold = config["rubric_generation"]["rrd_pairwise"]["redundancy_cosine_threshold"]
    for index, vector in enumerate(vectors):
        if any(float(np.dot(vector, vectors[kept_index])) > threshold for kept_index in kept_indices):
            continue
        kept_indices.append(index)
    return [criteria[index] for index in kept_indices]


def _cap_criteria(criteria: list[Criterion], *, config: dict[str, Any]) -> list[Criterion]:
    max_final = config["rubric_generation"]["rrd_pairwise"]["max_final_criteria"]
    if len(criteria) <= max_final:
        return criteria
    capped = sorted(criteria, key=lambda item: item.weight, reverse=True)[:max_final]
    total = sum(abs(item.weight) for item in capped) or 1.0
    for criterion in capped:
        criterion.weight = float(abs(criterion.weight) / total)
    return capped


def _assign_llm_weights(
    *,
    criteria: list[Criterion],
    prompt: str,
    reference_response: str | None,
    config: dict[str, Any],
    template_text: str,
    model_id: str,
) -> tuple[list[Criterion], dict[str, Any]]:
    if not criteria:
        return criteria, {"fallback_used": "no_criteria"}
    if config["experiment"]["smoke_test"]:
        total = sum(range(1, len(criteria) + 1))
        for index, criterion in enumerate(criteria, start=1):
            criterion.weight = float(index / total)
        return criteria, {"smoke_mock": True}

    rubric_lines = "\n".join(f"{index}. {criterion.text_ko}" for index, criterion in enumerate(criteria, start=1))
    prompt_text = render_prompt_template(
        template_text,
        prompt=prompt,
        reference_response=reference_response or "(none)",
        rubric_list=rubric_lines,
    )
    payload, raw_output = _call_openai_json(
        config=config,
        model=model_id,
        prompt=prompt_text,
        temperature=config["rubric_generation"]["llm_weighting"]["temperature"],
        max_output_tokens=config["rubric_generation"]["llm_weighting"]["max_output_tokens"],
        retries=2,
    )
    raw_weights = payload.get("weights")
    if not isinstance(raw_weights, list) or len(raw_weights) != len(criteria):
        raise ValueError("LLM weight assignment returned an invalid weights list.")
    normalized_weights: list[float] = []
    for item in raw_weights:
        value = float(item)
        normalized_weights.append(max(value, 0.0))
    total = sum(normalized_weights)
    if total <= 0:
        raise ValueError("LLM weight assignment returned a non-positive total weight.")
    for criterion, weight in zip(criteria, normalized_weights):
        criterion.weight = float(weight / total)
    return criteria, {"raw_output": raw_output, "raw_weights": raw_weights}


def _judge_generation_satisfaction(
    *,
    criterion: Criterion,
    prompt: str,
    response_text: str,
    reference_response: str | None,
    generator_model: str,
    config: dict[str, Any],
    smoke_test: bool,
) -> bool:
    if smoke_test:
        fingerprint = stable_hash(prompt, response_text, criterion.id, reference_response or "", length=2)
        return int(fingerprint, 16) % 2 == 0
    judge_prompt = """
당신은 평가 기준 충족 여부를 판단하는 판정자입니다.
반드시 JSON만 출력하세요.
{{"satisfied": true}}

<user_prompt>
{prompt}
</user_prompt>

<reference_response>
{reference_response}
</reference_response>

<criterion>
{criterion_json}
</criterion>

<model_response>
{response_text}
</model_response>
""".strip()
    payload, _ = _call_model_json(
        config=config,
        model_id=generator_model,
        prompt=judge_prompt.format(
            prompt=prompt,
            reference_response=reference_response or "(none)",
            criterion_json=criterion.model_dump_json(indent=2),
            response_text=response_text,
        ),
        max_new_tokens=128,
        temperature=0.0,
        retries=2,
        response_format_json=True,
    )
    return bool(payload.get("satisfied", False))


def _run_rrd_filter_prompt(
    *,
    criterion: Criterion,
    prompt: str,
    samples: list[AuxiliarySample],
    reference_response: str | None,
    generator_model: str,
    config: dict[str, Any],
    smoke_test: bool,
    template_text: str,
) -> tuple[bool, dict[str, Any]]:
    if smoke_test:
        return True, {"reason": "smoke_keep"}
    prompt_text = render_prompt_template(
        template_text,
        prompt=prompt,
        reference_response=reference_response or "(none)",
        criterion_json=criterion.model_dump_json(indent=2),
        sample_responses_json=_sample_responses_json(samples),
    )
    payload, raw_output = _call_model_json(
        config=config,
        model_id=generator_model,
        prompt=prompt_text,
        max_new_tokens=config["rubric_generation"]["rrd_pairwise"]["max_output_tokens"],
        temperature=config["rubric_generation"]["rrd_pairwise"]["temperature"],
        retries=2,
        response_format_json=True,
    )
    return bool(payload.get("keep", False)), {
        "reason": payload.get("reason", ""),
        "raw_output": raw_output,
    }


def _decompose_rrd_criterion(
    *,
    criterion: Criterion,
    prompt: str,
    samples: list[AuxiliarySample],
    reference_response: str | None,
    generator_model: str,
    config: dict[str, Any],
    smoke_test: bool,
    template_text: str,
) -> list[Criterion]:
    if smoke_test:
        return _mock_criteria(
            prompt_id=stable_hash(prompt, criterion.id, reference_response or ""),
            method="rrd_pairwise",
            count=2,
            prefix="D",
        )
    prompt_text = render_prompt_template(
        template_text,
        prompt=prompt,
        reference_response=reference_response or "(none)",
        criterion_json=criterion.model_dump_json(indent=2),
        sample_responses_json=_sample_responses_json(samples),
    )
    payload, _ = _call_model_json(
        config=config,
        model_id=generator_model,
        prompt=prompt_text,
        max_new_tokens=config["rubric_generation"]["rrd_pairwise"]["max_output_tokens"],
        temperature=config["rubric_generation"]["rrd_pairwise"]["temperature"],
        retries=2,
        response_format_json=True,
    )
    subcriteria = _criteria_from_payload(payload, key="subcriteria", prefix="D")
    if not 2 <= len(subcriteria) <= 4:
        raise ValueError(f"RRD decomposition expected 2-4 subcriteria, found {len(subcriteria)}")
    return subcriteria


def _expand_rrd_criterion(
    *,
    criterion: Criterion,
    prompt: str,
    samples: list[AuxiliarySample],
    reference_response: str | None,
    generator_model: str,
    config: dict[str, Any],
    depth: int,
    smoke_test: bool,
    filter_template: str,
    decompose_template: str,
    diagnostics: list[dict[str, Any]],
) -> list[Criterion]:
    keep, filter_meta = _run_rrd_filter_prompt(
        criterion=criterion,
        prompt=prompt,
        samples=samples,
        reference_response=reference_response,
        generator_model=generator_model,
        config=config,
        smoke_test=smoke_test,
        template_text=filter_template,
    )
    if not keep:
        diagnostics.append({"criterion_id": criterion.id, "depth": depth, "event": "filtered", **filter_meta})
        return []

    satisfaction = [
        int(
            _judge_generation_satisfaction(
                criterion=criterion,
                prompt=prompt,
                response_text=sample.response,
                reference_response=reference_response,
                generator_model=generator_model,
                config=config,
                smoke_test=smoke_test,
            )
        )
        for sample in samples
    ]
    count = int(sum(satisfaction))
    diagnostics.append({"criterion_id": criterion.id, "depth": depth, "event": "satisfaction", "count": count})

    threshold = config["rubric_generation"]["rrd_pairwise"]["decompose_if_satisfied_by_more_than"]
    if depth < config["rubric_generation"]["rrd_pairwise"]["max_depth"] and count > threshold:
        try:
            children = _decompose_rrd_criterion(
                criterion=criterion,
                prompt=prompt,
                samples=samples,
                reference_response=reference_response,
                generator_model=generator_model,
                config=config,
                smoke_test=smoke_test,
                template_text=decompose_template,
            )
            expanded: list[Criterion] = []
            for child_index, child in enumerate(children, start=1):
                child.id = f"{criterion.id}_{child_index}"
                expanded.extend(
                    _expand_rrd_criterion(
                        criterion=child,
                        prompt=prompt,
                        samples=samples,
                        reference_response=reference_response,
                        generator_model=generator_model,
                        config=config,
                        depth=depth + 1,
                        smoke_test=smoke_test,
                        filter_template=filter_template,
                        decompose_template=decompose_template,
                        diagnostics=diagnostics,
                    )
                )
            if expanded:
                diagnostics.append(
                    {
                        "criterion_id": criterion.id,
                        "depth": depth,
                        "event": "decomposed",
                        "child_count": len(expanded),
                    }
                )
                return expanded
        except Exception as exc:
            diagnostics.append({"criterion_id": criterion.id, "depth": depth, "event": "decompose_failed", "error": str(exc)})
    return [criterion]


def _fallback_rrd_criteria(*, prompt_id: str, method: str, max_count: int) -> list[Criterion]:
    criteria = _mock_criteria(prompt_id=prompt_id, method=method, count=min(max_count, 4), prefix="F")
    total = sum(item.weight for item in criteria) or 1.0
    for item in criteria:
        item.weight = float(item.weight / total)
    return criteria


def _method_requires_auxiliary_samples(method: str) -> bool:
    return method in {"rar_pairwise_sample", "rrd_pairwise_sample"}


def _generate_rrd_rubric(
    *,
    prompt_id: str,
    pair_id: str | None,
    prompt: str,
    samples: list[AuxiliarySample],
    reference_response: str | None,
    generator_family: str,
    generator_model: str,
    config: dict[str, Any],
    smoke_test: bool,
    initial_template: str,
    decompose_template: str,
    filter_template: str,
    weight_template: str,
    method: str,
) -> Rubric:
    reference_only_generation = method == "rrd_pairwise_reference"
    metadata: dict[str, Any] = {
        "fallback_used": None,
        "errors": [],
        "auxiliary_sample_ids": [sample.sample_id for sample in samples],
        "reference_guidance_used": bool(reference_response),
        "reference_only_generation": reference_only_generation,
    }
    try:
        if smoke_test:
            initial_criteria = _mock_criteria(
                prompt_id=stable_hash(prompt_id, pair_id or "", method),
                method=f"{method}_initial",
                count=config["rubric_generation"]["rrd_pairwise"]["initial_criteria"],
                prefix="R",
            )
            metadata["smoke_mock"] = True
        else:
            initial_prompt = render_prompt_template(
                initial_template,
                prompt=prompt,
                reference_response=reference_response or "(none)",
                sample_responses_json=_sample_responses_json(samples),
                initial_criteria_count=config["rubric_generation"]["rrd_pairwise"]["initial_criteria"],
            )
            payload, raw_output = _call_model_json(
                config=config,
                model_id=generator_model,
                prompt=initial_prompt,
                max_new_tokens=config["rubric_generation"]["rrd_pairwise"]["max_output_tokens"],
                temperature=config["rubric_generation"]["rrd_pairwise"]["temperature"],
                retries=2,
                response_format_json=True,
            )
            initial_criteria = _criteria_from_payload(
                payload,
                key="criteria",
                prefix="R",
                exact_count=config["rubric_generation"]["rrd_pairwise"]["initial_criteria"],
            )
            metadata["raw_output"] = raw_output

        diagnostics: list[dict[str, Any]] = []
        if reference_only_generation:
            expanded_criteria = list(initial_criteria)
            diagnostics.append(
                {
                    "event": "reference_only_generation",
                    "detail": "Skipped auxiliary-sample filter/decompose and used reference-only initial criteria.",
                }
            )
        else:
            expanded_criteria = []
            for criterion in initial_criteria:
                expanded_criteria.extend(
                    _expand_rrd_criterion(
                        criterion=criterion,
                        prompt=prompt,
                        samples=samples,
                        reference_response=reference_response,
                        generator_model=generator_model,
                        config=config,
                        depth=0,
                        smoke_test=smoke_test,
                        filter_template=filter_template,
                        decompose_template=decompose_template,
                        diagnostics=diagnostics,
                    )
                )
        expanded_criteria = _prune_redundant_criteria(expanded_criteria, config=config)
        if not expanded_criteria:
            raise RuntimeError("RRD left zero criteria after filtering/pruning.")

        expanded_criteria = _cap_criteria(expanded_criteria, config=config)
        weighted_criteria, weight_meta = _assign_llm_weights(
            criteria=expanded_criteria,
            prompt=prompt,
            reference_response=reference_response,
            config=config,
            template_text=weight_template,
            model_id=generator_model,
        )
        metadata["rrd_diagnostics"] = diagnostics
        metadata["weighting"] = weight_meta
        metadata["final_criterion_count"] = len(weighted_criteria)
        return Rubric(
            prompt_id=prompt_id,
            pair_id=pair_id,
            method=method,
            generator_family=generator_family,
            generator_model=generator_model,
            criteria=weighted_criteria,
            generation_metadata=metadata,
        )
    except Exception as exc:
        metadata["errors"].append(str(exc))
        metadata["fallback_used"] = "rrd_static_fallback"
        return Rubric(
            prompt_id=prompt_id,
            pair_id=pair_id,
            method=method,
            generator_family=generator_family,
            generator_model=generator_model,
            criteria=_fallback_rrd_criteria(
                prompt_id=stable_hash(prompt_id, pair_id or "", method),
                method=method,
                max_count=config["rubric_generation"]["rrd_pairwise"]["max_final_criteria"],
            ),
            generation_metadata=metadata,
        )


def _generate_rar_reference_rubric(
    *,
    pair: MetaEvalPair,
    generator_family: str,
    generator_model: str,
    config: dict[str, Any],
    smoke_test: bool,
    template_text: str,
) -> Rubric:
    metadata: dict[str, Any] = {"fallback_used": None, "errors": [], "reference_guidance_used": True}
    try:
        if smoke_test:
            items = _mock_rar_items(prompt_id=stable_hash(pair.prompt_id, pair.pair_id), method="rar_pairwise_reference")
            metadata["smoke_mock"] = True
        else:
            payload, raw_output = _call_model_json(
                config=config,
                model_id=generator_model,
                prompt=render_prompt_template(
                    template_text,
                    prompt=pair.prompt,
                    reference_response=_reference_response(pair),
                ),
                max_new_tokens=config["rubric_generation"]["rar_pairwise_reference"]["max_output_tokens"],
                temperature=config["rubric_generation"]["rar_pairwise_reference"]["temperature"],
                retries=2,
            )
            items = _parse_rar_items(
                payload,
                min_items=config["rubric_generation"]["rar_pairwise_reference"]["min_items"],
                max_items=config["rubric_generation"]["rar_pairwise_reference"]["max_items"],
            )
            metadata["raw_output"] = raw_output
        return Rubric(
            prompt_id=pair.prompt_id,
            pair_id=pair.pair_id,
            method="rar_pairwise_reference",
            generator_family=generator_family,
            generator_model=generator_model,
            rar_items=items,
            generation_metadata=metadata,
        )
    except Exception as exc:
        metadata["errors"].append(str(exc))
        metadata["fallback_used"] = "rar_static_fallback"
        return Rubric(
            prompt_id=pair.prompt_id,
            pair_id=pair.pair_id,
            method="rar_pairwise_reference",
            generator_family=generator_family,
            generator_model=generator_model,
            rar_items=_mock_rar_items(prompt_id=stable_hash(pair.prompt_id, pair.pair_id), method="rar_pairwise_reference"),
            generation_metadata=metadata,
        )


def _generate_rar_sample_rubric(
    *,
    prompt_id: str,
    prompt: str,
    samples: list[AuxiliarySample],
    generator_family: str,
    generator_model: str,
    config: dict[str, Any],
    smoke_test: bool,
    template_text: str,
) -> Rubric:
    metadata: dict[str, Any] = {
        "fallback_used": None,
        "errors": [],
        "auxiliary_sample_ids": [sample.sample_id for sample in samples],
        "reference_guidance_used": False,
    }
    try:
        if smoke_test:
            items = _mock_rar_items(prompt_id=prompt_id, method="rar_pairwise_sample")
            metadata["smoke_mock"] = True
        else:
            payload, raw_output = _call_model_json(
                config=config,
                model_id=generator_model,
                prompt=render_prompt_template(
                    template_text,
                    prompt=prompt,
                    sample_responses_json=_sample_responses_json(samples),
                ),
                max_new_tokens=config["rubric_generation"]["rar_pairwise_sample"]["max_output_tokens"],
                temperature=config["rubric_generation"]["rar_pairwise_sample"]["temperature"],
                retries=2,
            )
            items = _parse_rar_items(
                payload,
                min_items=config["rubric_generation"]["rar_pairwise_sample"]["min_items"],
                max_items=config["rubric_generation"]["rar_pairwise_sample"]["max_items"],
            )
            metadata["raw_output"] = raw_output
        return Rubric(
            prompt_id=prompt_id,
            pair_id=None,
            method="rar_pairwise_sample",
            generator_family=generator_family,
            generator_model=generator_model,
            rar_items=items,
            generation_metadata=metadata,
        )
    except Exception as exc:
        metadata["errors"].append(str(exc))
        metadata["fallback_used"] = "rar_static_fallback"
        return Rubric(
            prompt_id=prompt_id,
            pair_id=None,
            method="rar_pairwise_sample",
            generator_family=generator_family,
            generator_model=generator_model,
            rar_items=_mock_rar_items(prompt_id=prompt_id, method="rar_pairwise_sample"),
            generation_metadata=metadata,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--concurrency", type=int, default=None)
    parser.add_argument(
        "--method",
        choices=[
            "rar_pairwise_reference",
            "rar_pairwise_sample",
            "rrd_pairwise_reference",
            "rrd_pairwise_sample",
        ],
    )
    parser.add_argument("--generator_family")
    parser.add_argument("--prompt_limit", type=int)
    parser.add_argument("--prompt_id_file")
    args = parser.parse_args()

    config = load_config(args.config, smoke_test=args.smoke_test)
    all_pairs = load_meta_eval_pairs(get_meta_eval_pairs_path())
    prompt_rows = grouped_prompt_rows(all_pairs)
    prompt_id_allowlist: set[str] | None = None
    if args.prompt_id_file:
        prompt_id_path = Path(args.prompt_id_file)
        if not prompt_id_path.exists():
            raise RuntimeError(f"prompt_id_file not found: {prompt_id_path}")
        prompt_id_allowlist = {
            line.strip()
            for line in prompt_id_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        if not prompt_id_allowlist:
            raise RuntimeError(f"prompt_id_file is empty: {prompt_id_path}")
    samples_by_prompt: dict[str, list[AuxiliarySample]] = defaultdict(list)

    templates = {
        "rar_reference": load_prompt_template("configs/prompts/rar_generate_reference_ko.txt"),
        "rar_sample": load_prompt_template("configs/prompts/rar_generate_sample_ko.txt"),
        "rrd_initial_reference": load_prompt_template("configs/prompts/rrd_generate_initial_reference_ko.txt"),
        "rrd_initial_sample": load_prompt_template("configs/prompts/rrd_generate_initial_sample_ko.txt"),
        "rrd_decompose_reference": load_prompt_template("configs/prompts/rrd_decompose_reference_ko.txt"),
        "rrd_decompose_sample": load_prompt_template("configs/prompts/rrd_decompose_sample_ko.txt"),
        "rrd_filter_reference": load_prompt_template("configs/prompts/rrd_filter_reference_ko.txt"),
        "rrd_filter_sample": load_prompt_template("configs/prompts/rrd_filter_sample_ko.txt"),
        "rrd_llm_weighting": load_prompt_template("configs/prompts/rrd_llm_weighting_ko.txt"),
    }

    method_list = (
        [args.method]
        if args.method
        else [
            "rar_pairwise_reference",
            "rar_pairwise_sample",
            "rrd_pairwise_reference",
            "rrd_pairwise_sample",
        ]
    )
    if any(_method_requires_auxiliary_samples(method) for method in method_list):
        auxiliary_samples = load_auxiliary_samples(get_auxiliary_samples_path())
        for sample in auxiliary_samples:
            samples_by_prompt[sample.prompt_id].append(sample)
        for prompt_id, samples in samples_by_prompt.items():
            expected = config["auxiliary_response_generation"]["num_samples"]
            if len(samples) != expected:
                raise RuntimeError(f"Prompt {prompt_id} expected {expected} auxiliary samples, found {len(samples)}")
    generator_items = list(config["models"]["rubric_generators"].items())
    if args.generator_family:
        generator_items = [(family, model) for family, model in generator_items if family == args.generator_family]
        if not generator_items:
            raise RuntimeError(f"Unknown generator_family: {args.generator_family}")

    configured_concurrency = int(
        args.concurrency
        or config.get("runtime", {}).get("generation_concurrency", 1)
    )
    if configured_concurrency < 1:
        raise RuntimeError("generation concurrency must be >= 1")

    for generator_family, generator_model in generator_items:
        for method in method_list:
            output_path = get_rubric_output_path(method, generator_family)
            key_field = "pair_id" if method in {"rar_pairwise_reference", "rrd_pairwise_reference"} else "prompt_id"
            existing_rows = _load_output_rows(output_path, key_field=key_field)

            if method in {"rar_pairwise_reference", "rrd_pairwise_reference"}:
                expected_rows = len(all_pairs)
                completed = len(existing_rows)
                _write_progress(
                    output_path=output_path,
                    generator_family=generator_family,
                    method=method,
                    done=completed,
                    total=expected_rows,
                    key_field=key_field,
                )
                if completed:
                    print(f"[{generator_family}/{method}] resuming with {completed}/{expected_rows} completed")
                pending_pairs = [pair for pair in sorted(all_pairs, key=lambda item: (item.prompt_id, item.pair_id)) if pair.pair_id not in existing_rows]

                def build_pair_row(pair: MetaEvalPair) -> tuple[str, dict[str, Any]]:
                    if method == "rar_pairwise_reference":
                        rubric = _generate_rar_reference_rubric(
                            pair=pair,
                            generator_family=generator_family,
                            generator_model=generator_model,
                            config=config,
                            smoke_test=config["experiment"]["smoke_test"],
                            template_text=templates["rar_reference"],
                        )
                    else:
                        rubric = _generate_rrd_rubric(
                            prompt_id=pair.prompt_id,
                            pair_id=pair.pair_id,
                            prompt=pair.prompt,
                            samples=[],
                            reference_response=_reference_response(pair),
                            generator_family=generator_family,
                            generator_model=generator_model,
                            config=config,
                            smoke_test=config["experiment"]["smoke_test"],
                            initial_template=templates["rrd_initial_reference"],
                            decompose_template=templates["rrd_decompose_reference"],
                            filter_template=templates["rrd_filter_reference"],
                            weight_template=templates["rrd_llm_weighting"],
                            method="rrd_pairwise_reference",
                        )
                    return pair.pair_id, rubric.model_dump(mode="json")

                if configured_concurrency == 1:
                    iterator = ((None, build_pair_row(pair)) for pair in pending_pairs)
                else:
                    executor = ThreadPoolExecutor(max_workers=configured_concurrency)
                    future_map = {executor.submit(build_pair_row, pair): pair for pair in pending_pairs}
                    iterator = as_completed(future_map)

                try:
                    if configured_concurrency == 1:
                        for _, (row_key, row_payload) in iterator:
                            append_jsonl(output_path, [row_payload])
                            existing_rows[str(row_key)] = row_payload
                            completed += 1
                            if completed % 25 == 0 or completed == expected_rows:
                                print(f"[{generator_family}/{method}] {completed}/{expected_rows}")
                            if completed % 10 == 0 or completed == expected_rows:
                                _write_progress(
                                    output_path=output_path,
                                    generator_family=generator_family,
                                    method=method,
                                    done=completed,
                                    total=expected_rows,
                                    key_field=key_field,
                                )
                    else:
                        for future in iterator:
                            row_key, row_payload = future.result()
                            append_jsonl(output_path, [row_payload])
                            existing_rows[str(row_key)] = row_payload
                            completed += 1
                            if completed % 25 == 0 or completed == expected_rows:
                                print(f"[{generator_family}/{method}] {completed}/{expected_rows}")
                            if completed % 10 == 0 or completed == expected_rows:
                                _write_progress(
                                    output_path=output_path,
                                    generator_family=generator_family,
                                    method=method,
                                    done=completed,
                                    total=expected_rows,
                                    key_field=key_field,
                                )
                finally:
                    if configured_concurrency != 1:
                        executor.shutdown(wait=True, cancel_futures=False)
            else:
                prompt_items = sorted(prompt_rows.items())
                if prompt_id_allowlist is not None:
                    prompt_items = [(prompt_id, payload) for prompt_id, payload in prompt_items if prompt_id in prompt_id_allowlist]
                if args.prompt_limit is not None:
                    if args.prompt_limit < 1:
                        raise RuntimeError("--prompt_limit must be >= 1")
                    prompt_items = prompt_items[: args.prompt_limit]
                expected_rows = len(prompt_items)
                completed = len(existing_rows)
                _write_progress(
                    output_path=output_path,
                    generator_family=generator_family,
                    method=method,
                    done=completed,
                    total=expected_rows,
                    key_field=key_field,
                )
                if completed:
                    print(f"[{generator_family}/{method}] resuming with {completed}/{expected_rows} completed")
                pending_prompts = [(prompt_id, payload) for prompt_id, payload in prompt_items if prompt_id not in existing_rows]

                def build_prompt_row(prompt_id: str, payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
                    samples = sorted(samples_by_prompt.get(prompt_id, []), key=lambda item: item.sample_id)
                    if len(samples) != config["auxiliary_response_generation"]["num_samples"]:
                        raise RuntimeError(f"Prompt {prompt_id} missing auxiliary samples for {method}")
                    rrd_samples = _select_rrd_samples(samples, config=config) if method == "rrd_pairwise_sample" else samples
                    if method == "rar_pairwise_sample":
                        rubric = _generate_rar_sample_rubric(
                            prompt_id=prompt_id,
                            prompt=payload["prompt"],
                            samples=samples,
                            generator_family=generator_family,
                            generator_model=generator_model,
                            config=config,
                            smoke_test=config["experiment"]["smoke_test"],
                            template_text=templates["rar_sample"],
                        )
                    else:
                        rubric = _generate_rrd_rubric(
                            prompt_id=prompt_id,
                            pair_id=None,
                            prompt=payload["prompt"],
                            samples=rrd_samples,
                            reference_response=None,
                            generator_family=generator_family,
                            generator_model=generator_model,
                            config=config,
                            smoke_test=config["experiment"]["smoke_test"],
                            initial_template=templates["rrd_initial_sample"],
                            decompose_template=templates["rrd_decompose_sample"],
                            filter_template=templates["rrd_filter_sample"],
                            weight_template=templates["rrd_llm_weighting"],
                            method="rrd_pairwise_sample",
                        )
                    return prompt_id, rubric.model_dump(mode="json")

                if configured_concurrency == 1:
                    iterator = ((None, build_prompt_row(prompt_id, payload)) for prompt_id, payload in pending_prompts)
                else:
                    executor = ThreadPoolExecutor(max_workers=configured_concurrency)
                    future_map = {
                        executor.submit(build_prompt_row, prompt_id, payload): prompt_id
                        for prompt_id, payload in pending_prompts
                    }
                    iterator = as_completed(future_map)

                try:
                    if configured_concurrency == 1:
                        for _, (row_key, row_payload) in iterator:
                            append_jsonl(output_path, [row_payload])
                            existing_rows[str(row_key)] = row_payload
                            completed += 1
                            if completed % 25 == 0 or completed == expected_rows:
                                print(f"[{generator_family}/{method}] {completed}/{expected_rows}")
                            if completed % 10 == 0 or completed == expected_rows:
                                _write_progress(
                                    output_path=output_path,
                                    generator_family=generator_family,
                                    method=method,
                                    done=completed,
                                    total=expected_rows,
                                    key_field=key_field,
                                )
                    else:
                        for future in iterator:
                            row_key, row_payload = future.result()
                            append_jsonl(output_path, [row_payload])
                            existing_rows[str(row_key)] = row_payload
                            completed += 1
                            if completed % 25 == 0 or completed == expected_rows:
                                print(f"[{generator_family}/{method}] {completed}/{expected_rows}")
                            if completed % 10 == 0 or completed == expected_rows:
                                _write_progress(
                                    output_path=output_path,
                                    generator_family=generator_family,
                                    method=method,
                                    done=completed,
                                    total=expected_rows,
                                    key_field=key_field,
                                )
                finally:
                    if configured_concurrency != 1:
                        executor.shutdown(wait=True, cancel_futures=False)
            ordered_rows = [existing_rows[key] for key in sorted(existing_rows)]
            write_jsonl(output_path, ordered_rows)
            _write_progress(
                output_path=output_path,
                generator_family=generator_family,
                method=method,
                done=len(existing_rows),
                total=expected_rows,
                key_field=key_field,
            )
        unload_all_models()
    print("Wrote pairwise rubrics for full dataset")


if __name__ == "__main__":
    main()
