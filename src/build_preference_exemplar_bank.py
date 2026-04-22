"""Build the calibration-only preference exemplar bank."""

from __future__ import annotations

from .schemas import PreferenceExemplar
from .utils import (
    get_preference_exemplar_bank_path,
    get_split_pairs_path,
    load_config,
    load_meta_eval_pairs,
    parse_common_args,
    stable_hash,
    write_jsonl,
)


def main() -> None:
    args = parse_common_args()
    load_config(args.config, smoke_test=args.smoke_test)
    calibration_pairs = load_meta_eval_pairs(get_split_pairs_path("calibration"))
    exemplars: list[PreferenceExemplar] = []
    for pair in calibration_pairs:
        if pair.gold_preference == "tie":
            continue
        chosen = pair.response_a if pair.gold_preference == "A" else pair.response_b
        rejected = pair.response_b if pair.gold_preference == "A" else pair.response_a
        exemplars.append(
            PreferenceExemplar(
                exemplar_id=stable_hash(pair.prompt_id, pair.pair_id, prefix="ex"),
                prompt_id=pair.prompt_id,
                dataset=pair.dataset,
                prompt=pair.prompt,
                chosen=chosen,
                rejected=rejected,
                metadata={
                    "subset": pair.subset,
                    "source_pair_id": pair.pair_id,
                    **pair.metadata,
                },
            )
        )
    write_jsonl(get_preference_exemplar_bank_path(), [exemplar.model_dump(mode="json") for exemplar in exemplars])
    print(f"Wrote {len(exemplars)} preference exemplars to {get_preference_exemplar_bank_path()}")


if __name__ == "__main__":
    main()
