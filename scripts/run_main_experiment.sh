#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"

"$PYTHON_BIN" -m src.load_existing_human_eval --config configs/main.yaml
"$PYTHON_BIN" -m src.generate_auxiliary_samples --config configs/main.yaml
"$PYTHON_BIN" -m src.generate_rubrics --config configs/main.yaml
"$PYTHON_BIN" -m src.evaluate_pairs --config configs/main.yaml
"$PYTHON_BIN" -m src.export_run_manifest --config configs/main.yaml
