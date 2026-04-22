"""Evaluate non-tie pairwise rows with binary pairwise RaR / RRD protocols."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from .schemas import Criterion, PairPrediction, Rubric
from .utils import (
    build_rar_rubric_text,
    get_condition_specs,
    get_meta_eval_pairs_path,
    get_pair_predictions_path,
    get_rubric_output_path,
    load_config,
    load_meta_eval_pairs,
    load_prompt_template,
    parse_common_args,
    predict_pairwise_winner,
    read_jsonl,
    stable_hash,
    write_jsonl,
)
from .utils_openai import create_chat_completion, parse_json_response, preflight_chat_completion

_WINNER_PATTERNS = (
    r'(?i)"winner"\s*:\s*"(?P<winner>A|B)"',
    r"(?i)'winner'\s*:\s*'(?P<winner>A|B)'",
    r'(?i)\bwinner\b\s*[:=]\s*"?(?P<winner>A|B)"?\b',
)


@dataclass(frozen=True)
class BinaryEvalResult:
    passed: bool
    parse_failure: bool
    raw_output: str


def _build_task_context(prompt: str) -> str:
    return f"<user_prompt>\n{prompt}\n</user_prompt>"


def _parse_pairwise_judge_output(raw_text: str) -> tuple[str, str]:
    try:
        payload = parse_json_response(raw_text)
        if isinstance(payload, dict):
            winner = str(payload.get("winner", "")).strip().upper()
            justification = str(payload.get("justification", "")).strip()
            if winner in {"A", "B"}:
                return winner, justification or "[reason_missing]"
    except Exception:
        pass

    cleaned = " ".join(str(raw_text or "").replace("\\n", " ").split())
    for pattern in _WINNER_PATTERNS:
        match = re.search(pattern, cleaned)
        if match:
            return match.group("winner").upper(), "[regex-recovered]"
    raise ValueError("winner_must_be_A_or_B")


def _parse_yes_no(raw_text: str) -> bool:
    cleaned = str(raw_text or "").strip().upper()
    if cleaned == "YES":
        return True
    if cleaned == "NO":
        return False
    match = re.search(r"\b(YES|NO)\b", cleaned)
    if match:
        return match.group(1) == "YES"
    raise ValueError("response_must_contain_yes_or_no")


def _evaluate_pairwise_judge(
    *,
    config: dict[str, object],
    prompt_text: str,
    evaluator_model: str,
    smoke_test: bool,
    smoke_key: str,
) -> tuple[str, str, bool, str]:
    if smoke_test:
        winner = "A" if int(stable_hash(smoke_key, length=2), 16) % 2 == 0 else "B"
        justification = f"smoke heuristic winner={winner}"
        raw_output = json.dumps({"winner": winner, "justification": justification}, ensure_ascii=False)
        return winner, justification, False, raw_output

    raw_output = ""
    last_error: Exception | None = None
    retries = int(config["pairwise_evaluation"]["json_retry_limit"])
    for _ in range(retries):
        raw_output = create_chat_completion(
            config=config,
            model=evaluator_model,
            prompt=prompt_text,
            temperature=config["pairwise_evaluation"]["temperature"],
            max_output_tokens=config["pairwise_evaluation"]["max_output_tokens"],
            response_format_json=True,
        )
        try:
            winner, justification = _parse_pairwise_judge_output(raw_output)
            return winner, justification, False, raw_output
        except Exception as exc:
            last_error = exc
    fallback_winner = "A"
    fallback_reason = f"pairwise_parse_failure:{last_error}"
    return fallback_winner, fallback_reason, True, raw_output


def _evaluate_binary_criterion(
    *,
    config: dict[str, object],
    evaluator_model: str,
    prompt_text: str,
    smoke_test: bool,
    smoke_key: str,
) -> BinaryEvalResult:
    if smoke_test:
        passed = int(stable_hash(smoke_key, length=2), 16) % 2 == 0
        return BinaryEvalResult(passed=passed, parse_failure=False, raw_output="YES" if passed else "NO")

    raw_output = ""
    retries = int(config["binary_rubric_evaluation"]["json_retry_limit"])
    for _ in range(retries):
        raw_output = create_chat_completion(
            config=config,
            model=evaluator_model,
            prompt=prompt_text,
            temperature=config["binary_rubric_evaluation"]["temperature"],
            max_output_tokens=config["binary_rubric_evaluation"]["max_output_tokens"],
            response_format_json=False,
        )
        try:
            return BinaryEvalResult(passed=_parse_yes_no(raw_output), parse_failure=False, raw_output=raw_output)
        except Exception:
            continue
    return BinaryEvalResult(passed=False, parse_failure=True, raw_output=raw_output)


def _load_rubric_lookup(config: dict[str, object], split: str) -> dict[str, dict[str, Rubric]]:
    lookup: dict[str, dict[str, Rubric]] = {}
    for spec in get_condition_specs(config):
        if spec.rubric_scope == "none":
            continue
        assert spec.generator_family is not None
        path = get_rubric_output_path(spec.method, spec.generator_family)
        rows = [Rubric.model_validate(row) for row in read_jsonl(path)]
        if spec.rubric_scope == "pair":
            lookup[spec.condition_name] = {str(rubric.pair_id): rubric for rubric in rows if rubric.pair_id}
        else:
            lookup[spec.condition_name] = {rubric.prompt_id: rubric for rubric in rows}
    return lookup


def _score_response_with_rrd(
    *,
    config: dict[str, object],
    evaluator_model: str,
    prompt: str,
    response_text: str,
    criteria: list[Criterion],
    smoke_test: bool,
    cache_key_prefix: tuple[str, str, str],
    eval_template: str,
    eval_cache: dict[tuple[str, str, str, str], BinaryEvalResult],
) -> tuple[float, bool, int]:
    weighted_score = 0.0
    parse_failure = False
    passed_count = 0
    for criterion in criteria:
        cache_key = (*cache_key_prefix, criterion.id)
        cached = eval_cache.get(cache_key)
        if cached is None:
            prompt_text = eval_template.format(prompt=prompt, response=response_text, rubric=criterion.text_ko)
            cached = _evaluate_binary_criterion(
                config=config,
                evaluator_model=evaluator_model,
                prompt_text=prompt_text,
                smoke_test=smoke_test,
                smoke_key="||".join(cache_key),
            )
            eval_cache[cache_key] = cached
        parse_failure = parse_failure or cached.parse_failure
        if cached.passed:
            passed_count += 1
            weighted_score += float(criterion.weight)
    return weighted_score, parse_failure, passed_count


def main() -> None:
    args = parse_common_args()
    config = load_config(args.config, smoke_test=args.smoke_test)
    pairs = load_meta_eval_pairs(get_meta_eval_pairs_path())
    evaluator_model = str(config["models"]["evaluator_model"])

    baseline_template = load_prompt_template("configs/prompts/pairwise_baseline_judge_ko.txt")
    rubric_judge_template = load_prompt_template("configs/prompts/pairwise_rubric_judge_ko.txt")
    binary_eval_template = load_prompt_template("configs/prompts/rrd_binary_eval_ko.txt")

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
        )

    rubric_lookup = _load_rubric_lookup(config, "full")
    predictions: list[PairPrediction] = []
    binary_eval_cache: dict[tuple[str, str, str, str], BinaryEvalResult] = {}

    for pair in pairs:
        for spec in get_condition_specs(config):
            task_context = _build_task_context(pair.prompt)
            if spec.eval_protocol == "pairwise_judge":
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
                winner, justification, parse_failure, raw_output = _evaluate_pairwise_judge(
                    config=config,
                    prompt_text=prompt_text,
                    evaluator_model=evaluator_model,
                    smoke_test=config["experiment"]["smoke_test"],
                    smoke_key=f"{pair.pair_id}:{spec.condition_name}",
                )
                predictions.append(
                    PairPrediction(
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
                    )
                )
                continue

            rubric = rubric_lookup[spec.condition_name][pair.pair_id if spec.rubric_scope == "pair" else pair.prompt_id]
            scope_id = pair.pair_id if spec.rubric_scope == "pair" else pair.prompt_id
            score_a, parse_a, pass_count_a = _score_response_with_rrd(
                config=config,
                evaluator_model=evaluator_model,
                prompt=pair.prompt,
                response_text=pair.response_a,
                criteria=rubric.criteria,
                smoke_test=config["experiment"]["smoke_test"],
                cache_key_prefix=(spec.condition_name, scope_id, pair.response_a),
                eval_template=binary_eval_template,
                eval_cache=binary_eval_cache,
            )
            score_b, parse_b, pass_count_b = _score_response_with_rrd(
                config=config,
                evaluator_model=evaluator_model,
                prompt=pair.prompt,
                response_text=pair.response_b,
                criteria=rubric.criteria,
                smoke_test=config["experiment"]["smoke_test"],
                cache_key_prefix=(spec.condition_name, scope_id, pair.response_b),
                eval_template=binary_eval_template,
                eval_cache=binary_eval_cache,
            )
            predictions.append(
                PairPrediction(
                    pair_id=pair.pair_id,
                    method=spec.method,
                    generator_family=spec.generator_family,
                    generator_model=spec.generator_model,
                    eval_protocol="binary_rubric_aggregation",
                    pred_preference=predict_pairwise_winner(score_a, score_b, tie_breaker=str(config["experiment"]["pairwise_tie_breaker"])),
                    gold_preference=pair.gold_preference,  # type: ignore[arg-type]
                    score_a=score_a,
                    score_b=score_b,
                    parse_failure=parse_a or parse_b,
                    raw_output=json.dumps(
                        {
                            "criterion_count": len(rubric.criteria),
                            "pass_count_a": pass_count_a,
                            "pass_count_b": pass_count_b,
                        },
                        ensure_ascii=False,
                    ),
                )
            )

    write_jsonl(get_pair_predictions_path(), [prediction.model_dump(mode="json") for prediction in predictions])
    print(f"Wrote {len(predictions)} pairwise predictions for full dataset")


if __name__ == "__main__":
    main()
