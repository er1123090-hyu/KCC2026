"""Split the unified Korean meta-eval pairs into calibration and test sets."""

from __future__ import annotations

from .utils import (
    get_meta_eval_pairs_path,
    get_split_pairs_path,
    load_config,
    load_meta_eval_pairs,
    parse_common_args,
    split_pairs_by_prompt,
    write_jsonl,
)


def main() -> None:
    args = parse_common_args()
    config = load_config(args.config, smoke_test=args.smoke_test)
    pairs = load_meta_eval_pairs(get_meta_eval_pairs_path())
    calibration, test = split_pairs_by_prompt(
        pairs,
        calibration_ratio=config["splits"]["calibration_ratio"],
        seed=config["splits"]["seed"],
    )
    calibration_prompt_ids = {pair.prompt_id for pair in calibration}
    test_prompt_ids = {pair.prompt_id for pair in test}
    overlap = calibration_prompt_ids & test_prompt_ids
    if overlap:
        raise RuntimeError(f"Prompt leakage detected across splits: {sorted(overlap)[:5]}")
    write_jsonl(get_split_pairs_path("calibration"), [pair.model_dump(mode="json") for pair in calibration])
    write_jsonl(get_split_pairs_path("test"), [pair.model_dump(mode="json") for pair in test])
    print(
        f"Wrote {len(calibration)} calibration pairs and {len(test)} test pairs "
        f"to {get_split_pairs_path('calibration')} and {get_split_pairs_path('test')}"
    )


if __name__ == "__main__":
    main()

