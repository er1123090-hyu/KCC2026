"""Evaluate only pairwise-judge methods and persist progress incrementally."""

from __future__ import annotations

import json
from pathlib import Path

from .evaluate_pairs import _build_task_context, _evaluate_pairwise_judge
from .schemas import PairPrediction, Rubric
from .utils import (
    append_jsonl,
    build_rar_rubric_text,
    get_condition_specs,
    get_meta_eval_pairs_path,
    load_config,
    load_meta_eval_pairs,
    load_prompt_template,
    parse_common_args,
    read_jsonl,
    write_json,
)
from .utils_openai import preflight_chat_completion


def _output_path() -> Path:
    return Path("/data/minseo/KCC2026/results/raw/pair_predictions_pairwise_judge_only.jsonl")


def _progress_path() -> Path:
    return Path("/data/minseo/KCC2026/results/raw/pair_predictions_pairwise_judge_only.progress.json")


def _spec_key(method: str, generator_family: str | None) -> str:
    return method if generator_family is None else f"{method}__{generator_family}"


def _prioritize_specs(specs: list) -> list:
    priority = [
        ("pairwise_baseline", None),
        ("rar_pairwise_reference", "qwen_large"),
        ("rar_pairwise_sample", "qwen_large"),
        ("rar_pairwise_reference", "gemma_large"),
        ("rar_pairwise_sample", "gemma_large"),
    ]
    priority_rank = {item: index for index, item in enumerate(priority)}

    def sort_key(spec: object) -> tuple[int, str]:
        item = (spec.method, spec.generator_family)
        if item in priority_rank:
            return (0, f"{priority_rank[item]:04d}")
        return (1, _spec_key(spec.method, spec.generator_family))

    return sorted(specs, key=sort_key)


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


def _load_existing_rows() -> dict[tuple[str, str | None, str], dict[str, object]]:
    path = _output_path()
    existing: dict[tuple[str, str | None, str], dict[str, object]] = {}
    if not path.exists():
        return existing
    for row in read_jsonl(path):
        method = str(row.get("method", ""))
        generator_family = row.get("generator_family")
        pair_id = str(row.get("pair_id", ""))
        existing[(method, generator_family, pair_id)] = row
    return existing


def _write_progress(*, completed: int, total: int) -> None:
    write_json(
        _progress_path(),
        {
            "completed_rows": completed,
            "expected_rows": total,
            "remaining_rows": max(total - completed, 0),
            "complete": completed >= total,
        },
    )


def _is_resolved(row: dict[str, object] | None) -> bool:
    return row is not None and not bool(row.get("parse_failure"))


def _priority_group(*, row: dict[str, object] | None, generator_family: str | None) -> int:
    if row is not None and bool(row.get("parse_failure")):
        return 0
    if generator_family == "qwen_large":
        return 1
    if generator_family == "gemma_large":
        return 2
    return 3


def main() -> None:
    args = parse_common_args()
    config = load_config(args.config, smoke_test=args.smoke_test)
    pairs = load_meta_eval_pairs(get_meta_eval_pairs_path())
    evaluator_model = str(config["models"]["evaluator_model"])

    baseline_template = load_prompt_template("configs/prompts/pairwise_baseline_judge_ko.txt")
    rubric_judge_template = load_prompt_template("configs/prompts/pairwise_rubric_judge_ko.txt")

    if not config["experiment"]["smoke_test"]:
        preflight_chat_completion(
            config=config,
            model=evaluator_model,
            prompt=baseline_template.format(
                task_context="<user_prompt>\n준비 확인\n</user_prompt>",
                response_a="응답 A",
                response_b="응답 B",
            ),
            temperature=config["pairwise_evaluation"]["temperature"],
            max_output_tokens=64,
            reasoning_effort=config["pairwise_evaluation"].get("reasoning_effort"),
        )

    specs = _prioritize_specs([spec for spec in get_condition_specs(config) if spec.eval_protocol == "pairwise_judge"])
    rubric_lookup = _load_rubric_lookup(config)
    existing = _load_existing_rows()
    total = len(pairs) * len(specs)
    completed = sum(1 for row in existing.values() if _is_resolved(row))
    _write_progress(completed=completed, total=total)
    retried_keys: set[tuple[str, str | None, str]] = set()

    for spec in specs:
        spec_name = _spec_key(spec.method, spec.generator_family)
        spec_completed = sum(
            1 for pair in pairs if _is_resolved(existing.get((spec.method, spec.generator_family, pair.pair_id)))
        )
        spec_total = len(pairs)
        print(f"[pairwise_judge_only] starting {spec_name} ({spec_completed}/{spec_total} complete)")
        ordered_pairs = sorted(
            pairs,
            key=lambda pair: (
                _priority_group(
                    row=existing.get((spec.method, spec.generator_family, pair.pair_id)),
                    generator_family=spec.generator_family,
                ),
                pair.prompt_id,
                pair.pair_id,
            ),
        )
        for pair in ordered_pairs:
            key = (spec.method, spec.generator_family, pair.pair_id)
            if key in retried_keys:
                continue
            if _is_resolved(existing.get(key)):
                continue
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

            try:
                winner, justification, parse_failure, raw_output = _evaluate_pairwise_judge(
                    config=config,
                    prompt_text=prompt_text,
                    evaluator_model=evaluator_model,
                    smoke_test=config["experiment"]["smoke_test"],
                    smoke_key=f"{pair.pair_id}:{spec.condition_name}",
                )
            except Exception as exc:
                winner = "A"
                justification = f"evaluation_error:{exc}"
                parse_failure = True
                raw_output = f"[evaluation_error] {exc}"
            row = PairPrediction(
                pair_id=pair.pair_id,
                method=spec.method,
                generator_family=spec.generator_family,
                generator_model=spec.generator_model,
                eval_protocol="pairwise_judge",
                pred_preference=winner,
                gold_preference=pair.gold_preference,  # type: ignore[arg-type]
                justification=justification,
                parse_failure=parse_failure,
                raw_output=raw_output,
            ).model_dump(mode="json")
            append_jsonl(_output_path(), [row])
            was_resolved = _is_resolved(existing.get(key))
            existing[key] = row
            retried_keys.add(key)
            is_resolved = _is_resolved(row)
            if not was_resolved and is_resolved:
                completed += 1
                spec_completed += 1
            elif was_resolved and not is_resolved:
                completed = max(completed - 1, 0)
                spec_completed = max(spec_completed - 1, 0)
            if completed % 25 == 0 or spec_completed == spec_total or completed == total:
                print(
                    f"[pairwise_judge_only] total {completed}/{total} | "
                    f"{spec_name} {spec_completed}/{spec_total}"
                )
            _write_progress(completed=completed, total=total)
        print(f"[pairwise_judge_only] finished {spec_name} ({spec_completed}/{spec_total})")

    _write_progress(completed=completed, total=total)
    print(f"Wrote {completed} pairwise-judge rows to {_output_path()}")


if __name__ == "__main__":
    main()
