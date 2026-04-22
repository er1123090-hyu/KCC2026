"""Shared utilities for the Korean rubric grounding main experiment."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from dataclasses import dataclass
from hashlib import sha1
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import yaml

from .schemas import AuxiliarySample, Criterion, MetaEvalPair, PairPrediction, PreferenceExemplar, RaRRubricItem, Rubric

SMOKE_MARKER = "SMOKE_TEST_NOT_FOR_PAPER"


@dataclass(frozen=True)
class ConditionSpec:
    condition_name: str
    method: str
    generator_family: str | None
    generator_model: str | None
    eval_protocol: str
    rubric_scope: str


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_common_args(*, require_split: bool = False) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    if require_split:
        parser.add_argument("--split", required=True, choices=["calibration", "test"])
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--rrd_weighting_mode", choices=["llm", "uniform", "wu"])
    return parser.parse_args()


def load_config(config_path: str | Path, *, smoke_test: bool = False) -> dict[str, Any]:
    with Path(config_path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if smoke_test:
        config["experiment"]["smoke_test"] = True
    config["_project_root"] = str(project_root())
    return config


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def ensure_parent(path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def resolve_path(relative_path: str | Path) -> Path:
    return project_root() / Path(relative_path)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def stable_hash(*parts: Any, prefix: str | None = None, length: int = 12) -> str:
    joined = "||".join(str(part) for part in parts)
    digest = sha1(joined.encode("utf-8")).hexdigest()[:length]
    return f"{prefix}_{digest}" if prefix else digest


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalize_for_match(text: str) -> str:
    return normalize_whitespace(text)


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, payload: Any) -> Path:
    target = ensure_parent(path)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return target


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    target = Path(path)
    if not target.exists():
        return []
    rows: list[dict[str, Any]] = []
    with target.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> Path:
    target = ensure_parent(path)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return target


def append_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> Path:
    target = ensure_parent(path)
    with target.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return target


def load_prompt_template(relative_path: str) -> str:
    return resolve_path(relative_path).read_text(encoding="utf-8")


def render_prompt_template(template_text: str, **kwargs: Any) -> str:
    rendered = template_text
    for key, value in kwargs.items():
        rendered = rendered.replace("{" + key + "}", str(value))
    return rendered


def extract_first_json_object(raw_text: str) -> dict[str, Any]:
    raw_text = raw_text.strip()
    if raw_text.startswith("```"):
        raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
        raw_text = re.sub(r"\s*```$", "", raw_text)
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        pass

    start_positions = [index for index, char in enumerate(raw_text) if char == "{"]
    for start in start_positions:
        depth = 0
        for end in range(start, len(raw_text)):
            char = raw_text[end]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidate = raw_text[start : end + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        break
    raise ValueError("Could not locate a valid JSON object in model output.")


def normalize_preference(value: Any) -> str:
    if value is None:
        raise ValueError("Preference label is missing.")
    normalized = str(value).strip().lower()
    mapping = {
        "a": "A",
        "assistant_a": "A",
        "response_a": "A",
        "chosen": "A",
        "left": "A",
        "[[a]]": "A",
        "b": "B",
        "assistant_b": "B",
        "response_b": "B",
        "rejected": "B",
        "right": "B",
        "[[b]]": "B",
        "tie": "tie",
        "equal": "tie",
        "same": "tie",
        "draw": "tie",
        "both": "tie",
    }
    if normalized in mapping:
        return mapping[normalized]
    raise ValueError(f"Unsupported preference label: {value!r}")


def smoke_limit_meta_eval_pairs(pairs: Sequence[MetaEvalPair], *, max_pairs: int = 24) -> list[MetaEvalPair]:
    ordered_pairs = sorted(pairs, key=lambda item: (item.dataset, item.prompt_id, item.pair_id))
    selected: list[MetaEvalPair] = []
    seen_prompt_ids: set[str] = set()
    for pair in ordered_pairs:
        if len(selected) >= max_pairs:
            break
        if pair.prompt_id in seen_prompt_ids:
            continue
        selected.append(pair)
        seen_prompt_ids.add(pair.prompt_id)
    for pair in ordered_pairs:
        if len(selected) >= max_pairs:
            break
        if pair in selected:
            continue
        selected.append(pair)
    return selected


def split_pairs_by_prompt(
    pairs: Sequence[MetaEvalPair],
    *,
    calibration_ratio: float,
    seed: int,
) -> tuple[list[MetaEvalPair], list[MetaEvalPair]]:
    grouped: dict[str, list[MetaEvalPair]] = {}
    for pair in pairs:
        grouped.setdefault(pair.prompt_id, []).append(pair)
    prompt_ids = sorted(grouped)
    rng = random.Random(seed)
    rng.shuffle(prompt_ids)
    calibration_count = max(1, int(round(len(prompt_ids) * calibration_ratio))) if prompt_ids else 0
    calibration_prompt_ids = set(prompt_ids[:calibration_count])
    calibration = [pair for prompt_id in prompt_ids if prompt_id in calibration_prompt_ids for pair in grouped[prompt_id]]
    test = [pair for prompt_id in prompt_ids if prompt_id not in calibration_prompt_ids for pair in grouped[prompt_id]]
    return calibration, test


def get_processed_path(filename: str) -> Path:
    return resolve_path(Path("data/processed") / filename)


def get_results_path(filename: str) -> Path:
    return resolve_path(Path("results/raw") / filename)


def get_rrd_weighting_mode(config: dict[str, Any], cli_value: str | None = None) -> str:
    mode = (
        cli_value
        or os.environ.get("KCC_RRD_WEIGHTING_MODE")
        or config.get("binary_rubric_evaluation", {}).get("weighting_mode")
        or "llm"
    )
    normalized = str(mode).strip().lower()
    if normalized not in {"llm", "uniform", "wu"}:
        raise RuntimeError(f"Unsupported RRD weighting mode: {mode!r}")
    return normalized


def _results_suffix_for_weighting_mode(weighting_mode: str | None = None) -> str:
    normalized = (weighting_mode or "").strip().lower()
    if not normalized or normalized == "llm":
        return ""
    return f"_{normalized}"


def get_meta_eval_pairs_path() -> Path:
    return get_processed_path("meta_eval_pairs.jsonl")


def get_meta_eval_summary_path() -> Path:
    return get_processed_path("meta_eval_summary.json")


def get_split_pairs_path(split: str) -> Path:
    return get_processed_path(f"meta_eval_{split}.jsonl")


def get_preference_exemplar_bank_path() -> Path:
    return get_processed_path("preference_exemplar_bank.jsonl")


def get_auxiliary_samples_path(split: str | None = None) -> Path:
    if split is None:
        return get_processed_path("auxiliary_samples.jsonl")
    return get_processed_path(f"auxiliary_samples_{split}.jsonl")


def get_generic_rubric_path() -> Path:
    return get_processed_path("rubrics_generic.json")


def get_rubric_output_path(method: str, generator_family: str, split: str | None = None) -> Path:
    if split is None:
        return get_processed_path(f"rubrics_{method}__{generator_family}.jsonl")
    return get_processed_path(f"rubrics_{method}__{generator_family}__{split}.jsonl")


def get_judge_scores_path(split: str) -> Path:
    return get_results_path(f"judge_scores_{split}.jsonl")


def get_pair_predictions_path(split: str | None = None, *, weighting_mode: str | None = None) -> Path:
    suffix = _results_suffix_for_weighting_mode(weighting_mode)
    if split is None:
        return get_results_path(f"pair_predictions{suffix}.jsonl")
    return get_results_path(f"pair_predictions_{split}{suffix}.jsonl")


def get_thresholds_path() -> Path:
    return get_results_path("selected_tie_thresholds.json")


def get_run_manifest_path(*, weighting_mode: str | None = None) -> Path:
    suffix = _results_suffix_for_weighting_mode(weighting_mode)
    return get_results_path(f"run_manifest{suffix}.json")


def compose_condition_name(method: str, generator_family: str | None = None) -> str:
    return method if generator_family is None else f"{method}__{generator_family}"


def get_condition_specs(config: dict[str, Any]) -> list[ConditionSpec]:
    specs = [
        ConditionSpec(
            condition_name="pairwise_baseline",
            method="pairwise_baseline",
            generator_family=None,
            generator_model=None,
            eval_protocol="pairwise_judge",
            rubric_scope="none",
        )
    ]
    for generator_family, generator_model in config["models"]["rubric_generators"].items():
        specs.extend(
            [
                ConditionSpec(
                    condition_name=compose_condition_name("rar_pairwise_reference", generator_family),
                    method="rar_pairwise_reference",
                    generator_family=generator_family,
                    generator_model=generator_model,
                    eval_protocol="pairwise_judge",
                    rubric_scope="pair",
                ),
                ConditionSpec(
                    condition_name=compose_condition_name("rar_pairwise_sample", generator_family),
                    method="rar_pairwise_sample",
                    generator_family=generator_family,
                    generator_model=generator_model,
                    eval_protocol="pairwise_judge",
                    rubric_scope="prompt",
                ),
                ConditionSpec(
                    condition_name=compose_condition_name("rrd_pairwise_reference", generator_family),
                    method="rrd_pairwise_reference",
                    generator_family=generator_family,
                    generator_model=generator_model,
                    eval_protocol="binary_rubric_aggregation",
                    rubric_scope="pair",
                ),
                ConditionSpec(
                    condition_name=compose_condition_name("rrd_pairwise_sample", generator_family),
                    method="rrd_pairwise_sample",
                    generator_family=generator_family,
                    generator_model=generator_model,
                    eval_protocol="binary_rubric_aggregation",
                    rubric_scope="prompt",
                ),
            ]
        )
    method_allowlist = {
        item.strip()
        for item in os.environ.get("KCC_METHOD_ALLOWLIST", "").split(",")
        if item.strip()
    }
    condition_allowlist = {
        item.strip()
        for item in os.environ.get("KCC_CONDITION_ALLOWLIST", "").split(",")
        if item.strip()
    }
    if not method_allowlist and not condition_allowlist:
        return specs

    filtered = [
        spec
        for spec in specs
        if (not method_allowlist or spec.method in method_allowlist)
        and (not condition_allowlist or spec.condition_name in condition_allowlist)
    ]
    if not filtered:
        raise RuntimeError(
            "Condition filters removed every condition. "
            "Check KCC_METHOD_ALLOWLIST / KCC_CONDITION_ALLOWLIST."
        )
    return filtered


def load_meta_eval_pairs(path: str | Path) -> list[MetaEvalPair]:
    return [MetaEvalPair.model_validate(row) for row in read_jsonl(path)]


def load_preference_exemplars(path: str | Path) -> list[PreferenceExemplar]:
    return [PreferenceExemplar.model_validate(row) for row in read_jsonl(path)]


def load_auxiliary_samples(path: str | Path) -> list[AuxiliarySample]:
    return [AuxiliarySample.model_validate(row) for row in read_jsonl(path)]


def load_rubrics(path: str | Path) -> list[Rubric]:
    return [Rubric.model_validate(row) for row in read_jsonl(path)]


def detect_auxiliary_reuse(
    pairs: Sequence[MetaEvalPair],
    samples_by_prompt: dict[str, Sequence[AuxiliarySample]],
) -> None:
    for pair in pairs:
        pair_responses = {
            normalize_for_match(pair.response_a),
            normalize_for_match(pair.response_b),
        }
        for sample in samples_by_prompt.get(pair.prompt_id, []):
            if normalize_for_match(sample.response) in pair_responses:
                raise RuntimeError(
                    f"Leakage detected: auxiliary sample {sample.sample_id} for prompt_id={pair.prompt_id} "
                    f"matches an evaluated pair response."
                )


def predict_pairwise_winner(score_a: float, score_b: float, *, tie_breaker: str = "A") -> str:
    if score_a > score_b:
        return "A"
    if score_b > score_a:
        return "B"
    return tie_breaker


def rubric_as_json(criteria: Sequence[Criterion]) -> str:
    return json.dumps([criterion.model_dump(mode="json") for criterion in criteria], ensure_ascii=False, indent=2)


def build_rar_rubric_text(items: Sequence[RaRRubricItem]) -> str:
    lines: list[str] = []
    for index, item in enumerate(items, start=1):
        lines.append(f"{index}. {item.title}: {item.description} (weight={item.weight})")
    return "\n".join(lines)


def prepend_smoke_marker_to_readme() -> None:
    readme_path = resolve_path("README.md")
    if not readme_path.exists():
        return
    contents = readme_path.read_text(encoding="utf-8")
    if contents.startswith(SMOKE_MARKER):
        return
    readme_path.write_text(f"{SMOKE_MARKER}\n\n{contents}", encoding="utf-8")


def build_manifest_condition_key(method: str, generator_family: str | None) -> str:
    return compose_condition_name(method, generator_family)


def summarize_rubric_counts(rubrics: Sequence[Rubric]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for rubric in rubrics:
        key = build_manifest_condition_key(rubric.method, rubric.generator_family)
        counts[key] = counts.get(key, 0) + 1
    return counts


def collect_fallback_counts(rubrics: Sequence[Rubric]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for rubric in rubrics:
        fallback = rubric.generation_metadata.get("fallback_used")
        if fallback:
            counts[str(fallback)] = counts.get(str(fallback), 0) + 1
    return counts


def grouped_prompt_rows(pairs: Sequence[MetaEvalPair]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for pair in pairs:
        entry = grouped.setdefault(
            pair.prompt_id,
            {"prompt_id": pair.prompt_id, "prompt": pair.prompt, "pair_ids": []},
        )
        entry["pair_ids"].append(pair.pair_id)
    return grouped
