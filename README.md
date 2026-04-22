SMOKE_TEST_NOT_FOR_PAPER

# korean-rubric-grounding-main

This repo runs the binary pairwise raw experiment for Korean rubric-grounded evaluators.

## Goal

Run the main pairwise judge-quality experiment for Korean rubric-based evaluators, focused on judge construction rather than model superiority claims.

## Experiment Matrix

- Baseline
  - `pairwise_baseline`
- Rubric generator families
  - `qwen_small = Qwen/Qwen3-4B-FP8`
  - `qwen_large = Qwen/Qwen3-32B`
  - `gemma_small = google/gemma-4-E4B-it`
  - `gemma_large = google/gemma-4-31B-it`
- For each generator family
  - `rar_pairwise_reference__{generator}`
  - `rar_pairwise_sample__{generator}`
  - `rrd_pairwise_reference__{generator}`
  - `rrd_pairwise_sample__{generator}`
- Total judge conditions: `17`

## Data Policy

- Pairwise rows with human label `tie` are removed before evaluation.
- The experiment runs once over the full non-tie dataset instead of separate calibration/test splits.
- Sample-grounded methods use only the user prompt plus exactly 8 `gpt-5` auxiliary responses.
- Reference-grounded methods use the human-preferred response of the same pair as reference guidance.
- Exact score ties in binary rubric aggregation are broken as `A`, matching the original RRD paper-reproduction runner.

## Model Roles

- Auxiliary sample generator: `gpt-5`
- Fixed pairwise evaluator: `gpt-oss-120b`
- Rubric generators
  - `Qwen/Qwen3-4B-FP8`
  - `Qwen/Qwen3-32B`
  - `google/gemma-4-E4B-it`
  - `google/gemma-4-31B-it`

## Run

```bash
bash scripts/run_main_experiment.sh
```

For local HF inference on the requested devices:

```bash
CUDA_VISIBLE_DEVICES=0,1 bash scripts/run_main_experiment.sh
```

To automatically launch `openai/gpt-oss-120b` on GPUs `0,1,2,3` with vLLM after rubric generation finishes, then start pairwise evaluation:

```bash
bash scripts/watch_rubrics_then_vllm_pairwise_eval.sh
```

## Outputs

- `data/processed/meta_eval_pairs.jsonl`: unified non-tie pairwise human-eval rows
- `data/processed/meta_eval_summary.json`: raw-vs-filtered pair counts including dropped tie rows
- `data/processed/auxiliary_samples.jsonl`: exactly 8 auxiliary `gpt-5` samples per prompt
- `data/processed/rubrics_{method}__{generator}.jsonl`: pair-scoped or prompt-scoped rubric outputs
- `results/raw/pair_predictions.jsonl`: pair-level binary predictions
- `results/raw/run_manifest.json`: compact run manifest and fallback counts

## Exclusions

This repository intentionally excludes:

- statistical analysis
- plots
- final paper tables
- report generation
- stress evaluation
- RL training
- human annotation collection
