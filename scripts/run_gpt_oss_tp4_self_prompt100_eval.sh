#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
    DEFAULT_PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
else
    DEFAULT_PYTHON_BIN="$(command -v python3 || command -v python)"
fi

PYTHON_BIN="${PYTHON_BIN:-${DEFAULT_PYTHON_BIN}}"
VLLM_PYTHON_BIN="${VLLM_PYTHON_BIN:-/tmp/venvs/rrd-qwen-vllm/bin/python}"
CONFIG_PATH="${CONFIG_PATH:-configs/main.yaml}"
PROMPT_ID_FILE="${PROMPT_ID_FILE:-data/processed/rrd_sample_prompt_subset_100.ids.txt}"
RUBRIC_SPLIT="${RUBRIC_SPLIT:-prompt100}"
PREDICTION_SPLIT="${PREDICTION_SPLIT:-prompt100_self_gptoss120b_tp4}"
OPENAI_API_KEY_LOCAL="${OPENAI_API_KEY_LOCAL:-EMPTY}"

MODEL_NAME="${MODEL_NAME:-openai/gpt-oss-120b}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-gpt-oss-120b}"
PORT="${PORT:-8012}"
TP_SIZE="${TP_SIZE:-4}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.92}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
EVAL_CONCURRENCY="${EVAL_CONCURRENCY:-8}"

CONDITION_ALLOWLIST="${CONDITION_ALLOWLIST:-rar_pairwise_self__qwen_large,rrd_pairwise_self__qwen_large,rar_pairwise_self__gemma_large,rrd_pairwise_self__gemma_large}"

LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs}"
RUN_TAG="${RUN_TAG:-$(date '+%Y%m%d_%H%M%S')}"
RUN_LOG="${RUN_LOG:-${LOG_DIR}/gpt-oss-self-prompt100-eval-${RUN_TAG}.log}"
SERVER_LOG="${SERVER_LOG:-${LOG_DIR}/gpt-oss-self-prompt100-eval-${RUN_TAG}.server.log}"

mkdir -p "${LOG_DIR}"
: >>"${RUN_LOG}"
exec > >(tee -a "${RUN_LOG}") 2>&1

SERVER_PID=""

status() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

cleanup() {
    if [[ -n "${SERVER_PID}" ]]; then
        kill "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
        SERVER_PID=""
    fi
}
trap cleanup EXIT INT TERM

verify_prompt_subset() {
    "${PYTHON_BIN}" - <<PY
from pathlib import Path

path = Path(${PROMPT_ID_FILE@Q})
if not path.exists():
    raise SystemExit(f"Prompt id file not found: {path}")
ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
if len(ids) != 100:
    raise SystemExit(f"Expected 100 prompt ids in {path}, found {len(ids)}")
if len(set(ids)) != len(ids):
    raise SystemExit(f"Prompt id file contains duplicates: {path}")
print(f"Verified prompt subset: {len(ids)} prompt ids")
PY
}

wait_for_health() {
    local waited=0
    while (( waited < 3600 )); do
        if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
            status "gpt-oss TP${TP_SIZE} ready on port=${PORT}"
            return 0
        fi
        if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
            status "gpt-oss TP${TP_SIZE} exited during startup"
            tail -n 120 "${SERVER_LOG}" || true
            return 1
        fi
        sleep 5
        waited=$((waited + 5))
    done
    status "Timed out waiting for gpt-oss TP${TP_SIZE} health check"
    tail -n 120 "${SERVER_LOG}" || true
    return 1
}

verify_predictions() {
    "${PYTHON_BIN}" - <<PY
import json
from pathlib import Path

prompt_path = Path(${PROMPT_ID_FILE@Q})
prompt_ids = {
    line.strip()
    for line in prompt_path.read_text(encoding="utf-8").splitlines()
    if line.strip()
}
pair_path = Path("data/processed/meta_eval_pairs.jsonl")
pairs = [
    json.loads(line)
    for line in pair_path.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
pair_count = sum(1 for row in pairs if row["prompt_id"] in prompt_ids)
conditions = [item.strip() for item in ${CONDITION_ALLOWLIST@Q}.split(",") if item.strip()]
expected = pair_count * len(conditions)
pred_path = Path("results/raw") / f"pair_predictions_${PREDICTION_SPLIT}.jsonl"
if not pred_path.exists():
    raise SystemExit(f"Missing prediction file: {pred_path}")
actual = sum(1 for line in pred_path.open(encoding="utf-8") if line.strip())
if actual != expected:
    raise SystemExit(f"Expected {expected} predictions in {pred_path}, found {actual}")
print(f"Verified {pred_path.name}: {actual} rows ({pair_count} pairs x {len(conditions)} conditions)")
PY
}

status "Preparing gpt-oss TP${TP_SIZE} evaluation for prompt100 self rubrics"
status "Config=${CONFIG_PATH} rubric_split=${RUBRIC_SPLIT} prediction_split=${PREDICTION_SPLIT}"
status "Prompt ids=${PROMPT_ID_FILE}"
status "Conditions=${CONDITION_ALLOWLIST}"
status "Logs: run=${RUN_LOG} server=${SERVER_LOG}"
verify_prompt_subset

pkill -f "vllm.entrypoints.openai.api_server.*${PORT}" 2>/dev/null || true

status "Starting gpt-oss-120b on GPUs ${CUDA_DEVICES}"
CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" \
OPENAI_API_KEY="${OPENAI_API_KEY_LOCAL}" \
"${VLLM_PYTHON_BIN}" -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_NAME}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --port "${PORT}" \
    --tensor-parallel-size "${TP_SIZE}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --trust-remote-code \
    >"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

wait_for_health

status "Running evaluate_pairs with concurrency=${EVAL_CONCURRENCY}"
KCC_INCLUDE_SELF_CONDITIONS=1 \
KCC_CONDITION_ALLOWLIST="${CONDITION_ALLOWLIST}" \
OPENAI_BASE_URL="http://127.0.0.1:${PORT}/v1" \
OPENAI_API_KEY="${OPENAI_API_KEY_LOCAL}" \
"${PYTHON_BIN}" -m src.evaluate_pairs \
    --config "${CONFIG_PATH}" \
    --prompt-id-file "${PROMPT_ID_FILE}" \
    --rubric-split "${RUBRIC_SPLIT}" \
    --prediction-split "${PREDICTION_SPLIT}" \
    --concurrency "${EVAL_CONCURRENCY}"

verify_predictions
status "Completed prompt100 self rubric evaluation"
