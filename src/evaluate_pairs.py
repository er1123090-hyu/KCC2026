"""Evaluate non-tie pairwise rows with binary pairwise RaR / RRD protocols."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import re
from dataclasses import dataclass

import numpy as np

from .rrd_weighting import RrdWeightResult, apply_weight_result, compute_llm_weights, compute_uniform_weights, compute_wu_weights
from .schemas import AuxiliarySample, Criterion, MetaEvalPair, PairPrediction, Rubric
from .utils import (
    append_jsonl,
    build_rar_rubric_text,
    compose_condition_name,
    get_auxiliary_samples_path,
    get_condition_specs,
    get_meta_eval_pairs_path,
    get_pair_predictions_path,
    get_rrd_weighting_mode,
    get_rubric_output_path,
    load_auxiliary_samples,
    load_config,
    load_meta_eval_pairs,
    load_prompt_id_allowlist,
    load_prompt_template,
    parse_common_args,
    predict_pairwise_winner,
    read_jsonl,
    stable_hash,
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
    token = cleaned.strip().strip("\"'`.,:;()[]{}").upper()
    if token in {"A", "B"}:
        return token, "[letter-only]"
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
        try:
            raw_output = create_chat_completion(
                config=config,
                model=evaluator_model,
                prompt=prompt_text,
                temperature=config["pairwise_evaluation"]["temperature"],
                max_output_tokens=config["pairwise_evaluation"]["max_output_tokens"],
                reasoning_effort=config["pairwise_evaluation"].get("reasoning_effort"),
                response_format_json=True,
            )
        except Exception as exc:
            last_error = exc
            raw_output = f"[api_error] {exc}"
            continue
        try:
            winner, justification = _parse_pairwise_judge_output(raw_output)
            return winner, justification, False, raw_output
        except Exception as exc:
            last_error = exc

    strict_prompt = (
        prompt_text
        + "\n\n위 평가를 바탕으로 더 나은 응답을 고르세요. "
        + "최종 출력은 반드시 A 또는 B 한 글자만 쓰세요."
    )
    for _ in range(retries):
        try:
            raw_output = create_chat_completion(
                config=config,
                model=evaluator_model,
                prompt=strict_prompt,
                temperature=config["pairwise_evaluation"]["temperature"],
                max_output_tokens=max(256, min(512, int(config["pairwise_evaluation"].get("max_output_tokens", 256)))),
                reasoning_effort=config["pairwise_evaluation"].get("reasoning_effort"),
                response_format_json=False,
            )
        except Exception as exc:
            last_error = exc
            raw_output = f"[api_error] {exc}"
            continue
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
    last_error: Exception | None = None
    retries = int(config["binary_rubric_evaluation"]["json_retry_limit"])
    for _ in range(retries):
        try:
            raw_output = create_chat_completion(
                config=config,
                model=evaluator_model,
                prompt=prompt_text,
                temperature=config["binary_rubric_evaluation"]["temperature"],
                max_output_tokens=config["binary_rubric_evaluation"]["max_output_tokens"],
                reasoning_effort=config["binary_rubric_evaluation"].get("reasoning_effort"),
                response_format_json=False,
            )
        except Exception as exc:
            last_error = exc
            raw_output = f"[api_error] {exc}"
            continue
        try:
            return BinaryEvalResult(passed=_parse_yes_no(raw_output), parse_failure=False, raw_output=raw_output)
        except Exception:
            continue
    strict_prompt = (
        "You are a binary evaluator. Answer exactly one token: YES or NO. "
        "Do not explain.\n\n"
        + prompt_text
        + "\n\nFinal answer:"
    )
    for _ in range(retries):
        try:
            raw_output = create_chat_completion(
                config=config,
                model=evaluator_model,
                prompt=strict_prompt,
                temperature=config["binary_rubric_evaluation"]["temperature"],
                max_output_tokens=max(192, min(384, int(config["binary_rubric_evaluation"].get("max_output_tokens", 192)))),
                reasoning_effort=config["binary_rubric_evaluation"].get("reasoning_effort"),
                response_format_json=False,
            )
        except Exception as exc:
            last_error = exc
            raw_output = f"[api_error] {exc}"
            continue
        try:
            return BinaryEvalResult(passed=_parse_yes_no(raw_output), parse_failure=False, raw_output=raw_output)
        except Exception as exc:
            last_error = exc
    if last_error is not None and not raw_output:
        raw_output = f"[api_error] {last_error}"
    return BinaryEvalResult(passed=False, parse_failure=True, raw_output=raw_output)


def _load_rubric_lookup(config: dict[str, object], split: str) -> dict[str, dict[str, Rubric]]:
    lookup: dict[str, dict[str, Rubric]] = {}
    for spec in get_condition_specs(config):
        if spec.rubric_scope == "none":
            continue
        assert spec.generator_family is not None
        path = get_rubric_output_path(spec.method, spec.generator_family, split=split or None)
        rows = [Rubric.model_validate(row) for row in read_jsonl(path)]
        if spec.rubric_scope == "pair":
            lookup[spec.condition_name] = {str(rubric.pair_id): rubric for rubric in rows if rubric.pair_id}
        else:
            lookup[spec.condition_name] = {rubric.prompt_id: rubric for rubric in rows}
    return lookup


def _build_auxiliary_matrix(
    *,
    config: dict[str, object],
    evaluator_model: str,
    prompt: str,
    criteria: list[Criterion],
    sampled_responses: list[AuxiliarySample],
    smoke_test: bool,
    cache_key_prefix: tuple[str, str],
    eval_template: str,
    eval_cache: dict[tuple[str, str, str, str], BinaryEvalResult],
) -> tuple[np.ndarray, int]:
    matrix = np.zeros((len(sampled_responses), len(criteria)), dtype=int)
    parse_failure_count = 0
    for row_index, sample in enumerate(sampled_responses):
        for col_index, criterion in enumerate(criteria):
            cache_key = (*cache_key_prefix, sample.sample_id, criterion.id)
            cached = eval_cache.get(cache_key)
            if cached is None:
                prompt_text = eval_template.format(prompt=prompt, response=sample.response, rubric=criterion.text_ko)
                cached = _evaluate_binary_criterion(
                    config=config,
                    evaluator_model=evaluator_model,
                    prompt_text=prompt_text,
                    smoke_test=smoke_test,
                    smoke_key="||".join(cache_key),
                )
                eval_cache[cache_key] = cached
            matrix[row_index, col_index] = int(cached.passed)
            parse_failure_count += int(cached.parse_failure)
    return matrix, parse_failure_count


def _resolve_rrd_weight_result(
    *,
    config: dict[str, object],
    requested_mode: str,
    spec_condition_name: str,
    scope_id: str,
    prompt: str,
    criteria: list[Criterion],
    sampled_responses: list[AuxiliarySample],
    evaluator_model: str,
    smoke_test: bool,
    eval_template: str,
    eval_cache: dict[tuple[str, str, str, str], BinaryEvalResult],
    weight_cache: dict[tuple[str, str], RrdWeightResult],
) -> RrdWeightResult:
    cache_key = (spec_condition_name, scope_id)
    cached = weight_cache.get(cache_key)
    if cached is not None:
        return cached

    if requested_mode == "llm":
        result = compute_llm_weights(criteria)
        weight_cache[cache_key] = result
        return result
    if requested_mode == "uniform":
        result = compute_uniform_weights(criteria)
        weight_cache[cache_key] = result
        return result

    matrix, aux_parse_failure_count = _build_auxiliary_matrix(
        config=config,
        evaluator_model=evaluator_model,
        prompt=prompt,
        criteria=criteria,
        sampled_responses=sampled_responses,
        smoke_test=smoke_test,
        cache_key_prefix=cache_key,
        eval_template=eval_template,
        eval_cache=eval_cache,
    )
    result = compute_wu_weights(
        criteria,
        matrix,
        covariance_ridge=float(config["binary_rubric_evaluation"].get("wu_covariance_ridge", 1e-4)),
        min_covariance_samples=int(
            config["binary_rubric_evaluation"].get(
                "wu_min_covariance_samples",
                config["auxiliary_response_generation"]["num_samples"],
            )
        ),
        negative_weight_handling=str(
            config["binary_rubric_evaluation"].get("wu_negative_weight_handling", "clip_and_renorm")
        ),
    )
    diagnostics = dict(result.diagnostics)
    diagnostics["matrix_shape"] = list(matrix.shape)
    diagnostics["aux_parse_failure_count"] = aux_parse_failure_count
    result = RrdWeightResult(
        mode_requested=result.mode_requested,
        mode_used=result.mode_used,
        weights_by_id=result.weights_by_id,
        fallback_used=result.fallback_used,
        diagnostics=diagnostics,
    )
    weight_cache[cache_key] = result
    return result


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


def _evaluate_prediction(
    *,
    config: dict[str, object],
    pair: MetaEvalPair,
    spec,
    rubric_lookup: dict[str, dict[str, Rubric]],
    samples_by_prompt: dict[str, list[AuxiliarySample]],
    evaluator_model: str,
    weighting_modes: list[str],
    rubric_judge_template: str,
    baseline_template: str,
    binary_eval_template: str,
    eval_cache: dict[tuple[str, str, str, str], BinaryEvalResult],
    weight_caches: dict[str, dict[tuple[str, str], RrdWeightResult]],
) -> dict[str, tuple[tuple[str, str], dict[str, object]]]:
    prediction_key = (pair.pair_id, spec.condition_name)
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
        prediction = PairPrediction(
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
        return {weighting_modes[0]: (prediction_key, prediction.model_dump(mode="json"))}

    rubric = rubric_lookup[spec.condition_name][pair.pair_id if spec.rubric_scope == "pair" else pair.prompt_id]
    scope_id = pair.pair_id if spec.rubric_scope == "pair" else pair.prompt_id
    predictions_by_mode: dict[str, tuple[tuple[str, str], dict[str, object]]] = {}
    for weighting_mode in weighting_modes:
        weight_result = _resolve_rrd_weight_result(
            config=config,
            requested_mode=weighting_mode,
            spec_condition_name=spec.condition_name,
            scope_id=scope_id,
            prompt=pair.prompt,
            criteria=rubric.criteria,
            sampled_responses=samples_by_prompt.get(pair.prompt_id, []),
            evaluator_model=evaluator_model,
            smoke_test=config["experiment"]["smoke_test"],
            eval_template=binary_eval_template,
            eval_cache=eval_cache,
            weight_cache=weight_caches[weighting_mode],
        )
        weighted_criteria = apply_weight_result(rubric.criteria, weight_result)
        score_a, parse_a, pass_count_a = _score_response_with_rrd(
            config=config,
            evaluator_model=evaluator_model,
            prompt=pair.prompt,
            response_text=pair.response_a,
            criteria=weighted_criteria,
            smoke_test=config["experiment"]["smoke_test"],
            cache_key_prefix=(spec.condition_name, scope_id, pair.response_a),
            eval_template=binary_eval_template,
            eval_cache=eval_cache,
        )
        score_b, parse_b, pass_count_b = _score_response_with_rrd(
            config=config,
            evaluator_model=evaluator_model,
            prompt=pair.prompt,
            response_text=pair.response_b,
            criteria=weighted_criteria,
            smoke_test=config["experiment"]["smoke_test"],
            cache_key_prefix=(spec.condition_name, scope_id, pair.response_b),
            eval_template=binary_eval_template,
            eval_cache=eval_cache,
        )
        prediction = PairPrediction(
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
            weighting_mode_requested=weight_result.mode_requested,
            weighting_mode_used=weight_result.mode_used,
            weighting_fallback=weight_result.fallback_used,
            raw_output=json.dumps(
                {
                    "criterion_count": len(weighted_criteria),
                    "pass_count_a": pass_count_a,
                    "pass_count_b": pass_count_b,
                    "weighting_mode_requested": weight_result.mode_requested,
                    "weighting_mode_used": weight_result.mode_used,
                    "weighting_fallback": weight_result.fallback_used,
                    "weighting_diagnostics": weight_result.diagnostics,
                },
                ensure_ascii=False,
            ),
        )
        predictions_by_mode[weighting_mode] = (prediction_key, prediction.model_dump(mode="json"))
    return predictions_by_mode


def main() -> None:
    args = parse_common_args()
    config = load_config(args.config, smoke_test=args.smoke_test)
    pairs = load_meta_eval_pairs(get_meta_eval_pairs_path())
    prompt_id_allowlist = load_prompt_id_allowlist(args.prompt_id_file) if args.prompt_id_file else None
    if prompt_id_allowlist is not None:
        pairs = [pair for pair in pairs if pair.prompt_id in prompt_id_allowlist]
        if not pairs:
            raise RuntimeError("Prompt allowlist selection is empty for meta-eval pairs.")
    condition_specs = get_condition_specs(config)
    evaluator_model = str(config["models"]["evaluator_model"])
    requested_weighting_modes = (
        ["llm", "wu"] if args.rrd_weighting_mode == "both" else [get_rrd_weighting_mode(config, args.rrd_weighting_mode)]
    )
    if len(requested_weighting_modes) > 1 and any(spec.eval_protocol != "binary_rubric_aggregation" for spec in condition_specs):
        raise RuntimeError("rrd_weighting_mode=both requires only binary_rubric_aggregation conditions.")
    auxiliary_samples = load_auxiliary_samples(get_auxiliary_samples_path())
    if prompt_id_allowlist is not None:
        auxiliary_samples = [sample for sample in auxiliary_samples if sample.prompt_id in prompt_id_allowlist]
    samples_by_prompt: dict[str, list[AuxiliarySample]] = {}
    for sample in auxiliary_samples:
        samples_by_prompt.setdefault(sample.prompt_id, []).append(sample)
    for prompt_id in samples_by_prompt:
        samples_by_prompt[prompt_id].sort(key=lambda item: item.sample_id)
    configured_concurrency = int(
        args.concurrency
        or config.get("runtime", {}).get("evaluation_concurrency")
        or config.get("runtime", {}).get("generation_concurrency", 1)
    )
    if configured_concurrency < 1:
        raise RuntimeError("evaluation concurrency must be >= 1")

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
            reasoning_effort=config["pairwise_evaluation"].get("reasoning_effort"),
        )

    rubric_lookup = _load_rubric_lookup(config, args.rubric_split or "")
    active_condition_names = {spec.condition_name for spec in condition_specs}
    output_paths = {
        mode: get_pair_predictions_path(split=args.prediction_split or None, weighting_mode=mode)
        for mode in requested_weighting_modes
    }
    completed_prediction_keys_by_mode = {
        mode: {
            (str(row["pair_id"]), compose_condition_name(str(row["method"]), row.get("generator_family")))
            for row in read_jsonl(path)
            if compose_condition_name(str(row["method"]), row.get("generator_family")) in active_condition_names
        }
        for mode, path in output_paths.items()
    }
    total_predictions = len(pairs) * len(condition_specs)
    remaining_predictions_by_mode = {
        mode: total_predictions - len(completed_prediction_keys_by_mode[mode])
        for mode in requested_weighting_modes
    }
    print(
        "Starting pairwise evaluation "
        f"pairs={len(pairs)} "
        f"conditions={len(condition_specs)} "
        f"rubric_split={args.rubric_split or '[default]'} "
        f"prediction_split={args.prediction_split or '[default]'} "
        f"concurrency={configured_concurrency} "
        f"weighting_modes={','.join(requested_weighting_modes)} "
        f"remaining="
        + ",".join(f"{mode}:{remaining_predictions_by_mode[mode]}" for mode in requested_weighting_modes),
        flush=True,
    )
    if all(remaining <= 0 for remaining in remaining_predictions_by_mode.values()):
        print("No remaining predictions for the active condition set.", flush=True)
        return

    newly_written_by_mode = {mode: 0 for mode in requested_weighting_modes}
    next_progress_mark_by_mode = {mode: 100 for mode in requested_weighting_modes}
    shared_eval_cache: dict[tuple[str, str, str, str], BinaryEvalResult] = {}
    shared_weight_caches = {mode: {} for mode in requested_weighting_modes}
    jobs = [
        (
            pair,
            spec,
            [
                mode
                for mode in requested_weighting_modes
                if (pair.pair_id, spec.condition_name) not in completed_prediction_keys_by_mode[mode]
            ],
        )
        for pair in pairs
        for spec in condition_specs
        if any(
            (pair.pair_id, spec.condition_name) not in completed_prediction_keys_by_mode[mode]
            for mode in requested_weighting_modes
        )
    ]
    write_buffers = {mode: [] for mode in requested_weighting_modes}
    flush_every = max(1, min(configured_concurrency, 20))

    def flush_buffer(mode: str | None = None) -> None:
        if mode is None:
            for item in requested_weighting_modes:
                flush_buffer(item)
            return
        if write_buffers[mode]:
            append_jsonl(output_paths[mode], write_buffers[mode])
            write_buffers[mode] = []

    if configured_concurrency == 1:
        for pair, spec, pending_modes in jobs:
            predictions_by_mode = _evaluate_prediction(
                config=config,
                pair=pair,
                spec=spec,
                rubric_lookup=rubric_lookup,
                samples_by_prompt=samples_by_prompt,
                evaluator_model=evaluator_model,
                weighting_modes=pending_modes,
                rubric_judge_template=rubric_judge_template,
                baseline_template=baseline_template,
                binary_eval_template=binary_eval_template,
                eval_cache=shared_eval_cache,
                weight_caches=shared_weight_caches,
            )
            for mode, (prediction_key, row) in predictions_by_mode.items():
                write_buffers[mode].append(row)
                completed_prediction_keys_by_mode[mode].add(prediction_key)
                newly_written_by_mode[mode] += 1
                if len(write_buffers[mode]) >= flush_every:
                    flush_buffer(mode)
                while newly_written_by_mode[mode] >= next_progress_mark_by_mode[mode]:
                    print(
                        "Progress "
                        f"mode={mode} "
                        f"newly_written={newly_written_by_mode[mode]}/{remaining_predictions_by_mode[mode]} "
                        f"total_output={len(completed_prediction_keys_by_mode[mode])}/{total_predictions} "
                        f"last_pair={pair.pair_id}",
                        flush=True,
                    )
                    next_progress_mark_by_mode[mode] += 100
    else:
        with ThreadPoolExecutor(max_workers=configured_concurrency) as executor:
            future_to_pair = {
                executor.submit(
                    _evaluate_prediction,
                    config=config,
                    pair=pair,
                    spec=spec,
                    rubric_lookup=rubric_lookup,
                    samples_by_prompt=samples_by_prompt,
                    evaluator_model=evaluator_model,
                    weighting_modes=pending_modes,
                    rubric_judge_template=rubric_judge_template,
                    baseline_template=baseline_template,
                    binary_eval_template=binary_eval_template,
                    eval_cache=shared_eval_cache,
                    weight_caches=shared_weight_caches,
                ): pair
                for pair, spec, pending_modes in jobs
            }
            for future in as_completed(future_to_pair):
                predictions_by_mode = future.result()
                for mode, (prediction_key, row) in predictions_by_mode.items():
                    write_buffers[mode].append(row)
                    completed_prediction_keys_by_mode[mode].add(prediction_key)
                    newly_written_by_mode[mode] += 1
                    if len(write_buffers[mode]) >= flush_every:
                        flush_buffer(mode)
                    while newly_written_by_mode[mode] >= next_progress_mark_by_mode[mode]:
                        print(
                            "Progress "
                            f"mode={mode} "
                            f"newly_written={newly_written_by_mode[mode]}/{remaining_predictions_by_mode[mode]} "
                            f"total_output={len(completed_prediction_keys_by_mode[mode])}/{total_predictions} "
                            f"last_pair={future_to_pair[future].pair_id}",
                            flush=True,
                        )
                        next_progress_mark_by_mode[mode] += 100

    flush_buffer()

    print(
        "Wrote new pairwise predictions for active selection "
        + ", ".join(
            f"{mode}={newly_written_by_mode[mode]}"
            for mode in requested_weighting_modes
        )
        + f" (active conditions={len(condition_specs)})",
        flush=True,
    )


if __name__ == "__main__":
    main()
