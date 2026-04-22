"""Export a compact run manifest for the binary pairwise experiment."""

from __future__ import annotations

import subprocess
from collections import defaultdict
from datetime import datetime, timezone

from .schemas import PairPrediction, Rubric
from .utils import (
    SMOKE_MARKER,
    collect_fallback_counts,
    compose_condition_name,
    get_auxiliary_samples_path,
    get_condition_specs,
    get_meta_eval_pairs_path,
    get_meta_eval_summary_path,
    get_pair_predictions_path,
    get_rubric_output_path,
    get_run_manifest_path,
    load_auxiliary_samples,
    load_config,
    load_meta_eval_pairs,
    parse_common_args,
    prepend_smoke_marker_to_readme,
    read_json,
    read_jsonl,
    summarize_rubric_counts,
    write_json,
)


def _git_hash() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def main() -> None:
    args = parse_common_args()
    config = load_config(args.config, smoke_test=args.smoke_test)
    meta_pairs = load_meta_eval_pairs(get_meta_eval_pairs_path())
    meta_summary = read_json(get_meta_eval_summary_path()) if get_meta_eval_summary_path().exists() else {}
    all_samples = load_auxiliary_samples(get_auxiliary_samples_path())

    rubrics: list[Rubric] = []
    for spec in get_condition_specs(config):
        if spec.rubric_scope == "none":
            continue
        assert spec.generator_family is not None
        rubrics.extend(Rubric.model_validate(row) for row in read_jsonl(get_rubric_output_path(spec.method, spec.generator_family)))

    predictions = [PairPrediction.model_validate(row) for row in read_jsonl(get_pair_predictions_path())]
    prediction_counts: dict[str, int] = defaultdict(int)
    for prediction in predictions:
        prediction_counts[compose_condition_name(prediction.method, prediction.generator_family)] += 1

    experiment_name = config["experiment"]["name"]
    if config["experiment"]["smoke_test"]:
        experiment_name = f"{SMOKE_MARKER} {experiment_name}"
        prepend_smoke_marker_to_readme()

    rubric_counts = summarize_rubric_counts(rubrics)
    rubric_counts.setdefault("pairwise_baseline", 0)

    manifest = {
        "experiment_name": experiment_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_hash": _git_hash(),
        "models": {
            "evaluator_model": config["models"]["evaluator_model"],
            "auxiliary_sample_generator": config["auxiliary_response_generation"]["model"],
            "rubric_generators": config["models"]["rubric_generators"],
        },
        "counts": {
            "total_meta_eval_pairs": len(meta_pairs),
            "total_raw_pairs_before_tie_filter": meta_summary.get("total_raw_pairs"),
            "dropped_tie_pairs": meta_summary.get("dropped_tie_pair_count", 0),
            "non_tie_pairs": len(meta_pairs),
            "unique_prompts": len({pair.prompt_id for pair in meta_pairs}),
            "auxiliary_samples_generated": len(all_samples),
            "rubrics_generated_by_condition": rubric_counts,
            "pair_predictions_written_by_condition": dict(prediction_counts),
        },
        "fallback_counts": collect_fallback_counts(rubrics),
    }
    write_json(get_run_manifest_path(), manifest)
    print(f"Wrote run manifest to {get_run_manifest_path()}")


if __name__ == "__main__":
    main()
