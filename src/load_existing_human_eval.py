"""Load existing Korean human-eval pairwise datasets into a unified schema."""

from __future__ import annotations

from typing import Any

from .schemas import MetaEvalPair
from .utils import (
    get_meta_eval_pairs_path,
    get_meta_eval_summary_path,
    load_config,
    normalize_preference,
    parse_common_args,
    smoke_limit_meta_eval_pairs,
    stable_hash,
    write_json,
    write_jsonl,
)

PROMPT_CANDIDATES = ["prompt", "question", "instruction", "query", "user_prompt", "judge_query"]
RESPONSE_A_CANDIDATES = [
    "response_a",
    "answer_a",
    "chosen",
    "chosen_response",
    "output_a",
    "model_a_response",
    "A",
]
RESPONSE_B_CANDIDATES = [
    "response_b",
    "answer_b",
    "rejected",
    "rejected_response",
    "output_b",
    "model_b_response",
    "B",
]
PREFERENCE_CANDIDATES = ["preference", "human_preference", "winner", "label", "gold", "decision"]


def _import_datasets():
    from datasets import get_dataset_config_names, load_dataset, load_dataset_builder

    return get_dataset_config_names, load_dataset, load_dataset_builder


def _first_existing(columns: list[str], candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def _print_schema_and_fail(dataset_name: str, configs: list[str], details: list[dict[str, Any]], columns: list[str]) -> None:
    print(f"Dataset schema unknown for {dataset_name}")
    print(f"Configs: {configs}")
    print(f"Splits: {details}")
    print(f"Columns: {columns}")
    raise RuntimeError(f"Unsupported schema for dataset: {dataset_name}")


def _normalize_row(
    *,
    dataset_name: str,
    subset_name: str | None,
    row: dict[str, Any],
    row_index: int,
    prompt_key: str,
    response_a_key: str,
    response_b_key: str,
    preference_key: str,
) -> MetaEvalPair:
    prompt = str(row[prompt_key]).strip()
    response_a = str(row[response_a_key]).strip()
    response_b = str(row[response_b_key]).strip()
    preference = normalize_preference(row[preference_key])
    prompt_id = stable_hash(prompt, prefix="prompt")
    pair_id = stable_hash(dataset_name, subset_name, row_index, prompt, response_a, response_b, prefix="pair")
    metadata = {
        "row_index": row_index,
        "prompt_key": prompt_key,
        "response_a_key": response_a_key,
        "response_b_key": response_b_key,
        "preference_key": preference_key,
    }
    for extra_key in row.keys():
        if extra_key not in {prompt_key, response_a_key, response_b_key, preference_key}:
            metadata[extra_key] = row[extra_key]
    return MetaEvalPair(
        pair_id=pair_id,
        prompt_id=prompt_id,
        dataset=dataset_name,
        subset=subset_name,
        prompt=prompt,
        response_a=response_a,
        response_b=response_b,
        gold_preference=preference,
        metadata=metadata,
    )


def _normalize_pairwise_false_row(
    *,
    dataset_name: str,
    subset_name: str | None,
    row: dict[str, Any],
    row_index: int,
) -> MetaEvalPair:
    prompt = str(row["instruction"]).strip()
    response_a = str(row["response_with_false_info"]).strip()
    response_b = str(row["original_response"]).strip()
    preference = normalize_preference(row["winner"])
    prompt_id = stable_hash(prompt, prefix="prompt")
    pair_id = stable_hash(dataset_name, subset_name, row_index, prompt, response_a, response_b, prefix="pair")
    metadata = {
        "row_index": row_index,
        "prompt_key": "instruction",
        "response_a_key": "response_with_false_info",
        "response_b_key": "original_response",
        "preference_key": "winner",
    }
    for extra_key, extra_value in row.items():
        if extra_key not in {"instruction", "response_with_false_info", "original_response", "winner"}:
            metadata[extra_key] = extra_value
    return MetaEvalPair(
        pair_id=pair_id,
        prompt_id=prompt_id,
        dataset=dataset_name,
        subset=subset_name,
        prompt=prompt,
        response_a=response_a,
        response_b=response_b,
        gold_preference=preference,
        metadata=metadata,
    )


def _load_dataset_rows(config: dict[str, Any]) -> list[MetaEvalPair]:
    get_dataset_config_names, load_dataset, load_dataset_builder = _import_datasets()
    normalized_rows: list[MetaEvalPair] = []

    for dataset_spec in config["data"]["meta_eval_datasets"]:
        dataset_name = dataset_spec["dataset_name"]
        configs = list(get_dataset_config_names(dataset_name))
        builder_details: list[dict[str, Any]] = []
        pairwise_configs: list[tuple[str, list[str]]] = []

        for config_name in configs:
            builder = load_dataset_builder(dataset_name, config_name)
            splits = list((builder.info.splits or {}).keys())
            columns = list((builder.info.features or {}).keys())
            builder_details.append({"config": config_name, "splits": splits, "columns": columns})
            if dataset_spec.get("use_pairwise_like_subsets_only"):
                is_pairwise = "pairwise" in config_name.lower() or config_name in {"Pairwise", "Pairwise-False"}
                include_false = dataset_spec.get("include_false_subset_if_available", False)
                if not is_pairwise:
                    continue
                if "false" in config_name.lower() and not include_false:
                    continue
            pairwise_configs.append((config_name, splits))

        if dataset_name == "HAERAE-HUB/Korean-Human-Judgements":
            pairwise_configs = [(details["config"], details["splits"]) for details in builder_details]

        if not pairwise_configs:
            _print_schema_and_fail(dataset_name, configs, builder_details, [])

        for config_name, splits in pairwise_configs:
            if not splits:
                continue
            split_name = splits[0]
            dataset_rows = load_dataset(dataset_name, name=config_name, split=split_name)
            columns = list(dataset_rows.column_names)

            if {"instruction", "response_with_false_info", "original_response", "winner"}.issubset(columns):
                for row_index, row in enumerate(dataset_rows):
                    normalized_rows.append(
                        _normalize_pairwise_false_row(
                            dataset_name=dataset_name,
                            subset_name=config_name,
                            row=row,
                            row_index=row_index,
                        )
                    )
                continue

            prompt_key = _first_existing(columns, PROMPT_CANDIDATES)
            response_a_key = _first_existing(columns, RESPONSE_A_CANDIDATES)
            response_b_key = _first_existing(columns, RESPONSE_B_CANDIDATES)
            preference_key = _first_existing(columns, PREFERENCE_CANDIDATES)

            if None in {prompt_key, response_a_key, response_b_key, preference_key}:
                _print_schema_and_fail(dataset_name, configs, builder_details, columns)

            for row_index, row in enumerate(dataset_rows):
                normalized_rows.append(
                    _normalize_row(
                        dataset_name=dataset_name,
                        subset_name=config_name,
                        row=row,
                        row_index=row_index,
                        prompt_key=str(prompt_key),
                        response_a_key=str(response_a_key),
                        response_b_key=str(response_b_key),
                        preference_key=str(preference_key),
                    )
                )
    return normalized_rows


def main() -> None:
    args = parse_common_args()
    config = load_config(args.config, smoke_test=args.smoke_test)
    all_rows = _load_dataset_rows(config)
    dropped_tie_rows = [row for row in all_rows if row.gold_preference == "tie"]
    rows = [row for row in all_rows if row.gold_preference != "tie"]
    if config["experiment"]["smoke_test"]:
        rows = smoke_limit_meta_eval_pairs(rows)
    write_jsonl(get_meta_eval_pairs_path(), [row.model_dump(mode="json") for row in rows])
    write_json(
        get_meta_eval_summary_path(),
        {
            "total_raw_pairs": len(all_rows),
            "dropped_tie_pair_count": len(dropped_tie_rows),
            "written_non_tie_pair_count": len(rows),
            "dropped_tie_prompt_count": len({row.prompt_id for row in dropped_tie_rows}),
        },
    )
    print(
        f"Wrote {len(rows)} normalized non-tie pairs to {get_meta_eval_pairs_path()} "
        f"(dropped {len(dropped_tie_rows)} ties)"
    )


if __name__ == "__main__":
    main()
