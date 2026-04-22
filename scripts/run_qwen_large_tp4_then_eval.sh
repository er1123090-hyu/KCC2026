#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
VLLM_PYTHON_BIN="${VLLM_PYTHON_BIN:-/tmp/venvs/rrd-qwen-vllm/bin/python}"
EVAL_VLLM_PYTHON_BIN="${EVAL_VLLM_PYTHON_BIN:-/tmp/venvs/rrd-qwen-vllm/bin/python}"
OPENAI_API_KEY_LOCAL="${OPENAI_API_KEY_LOCAL:-EMPTY}"
QWEN_MODEL_NAME="${QWEN_MODEL_NAME:-Qwen/Qwen3-32B}"
QWEN_SERVED_NAME="${QWEN_SERVED_NAME:-Qwen/Qwen3-32B}"
QWEN_PORT="${QWEN_PORT:-8106}"
QWEN_TP_SIZE="${QWEN_TP_SIZE:-4}"
QWEN_GPU_MEMORY_UTILIZATION="${QWEN_GPU_MEMORY_UTILIZATION:-0.90}"
QWEN_MAX_MODEL_LEN="${QWEN_MAX_MODEL_LEN:-32768}"
QWEN_CONCURRENCY="${QWEN_CONCURRENCY:-30}"

EVAL_MODEL_NAME="${EVAL_MODEL_NAME:-openai/gpt-oss-120b}"
EVAL_SERVED_NAME="${EVAL_SERVED_NAME:-gpt-oss-120b}"
EVAL_PORT="${EVAL_PORT:-8012}"
EVAL_TP_SIZE="${EVAL_TP_SIZE:-4}"
EVAL_GPU_MEMORY_UTILIZATION="${EVAL_GPU_MEMORY_UTILIZATION:-0.92}"
EVAL_MAX_MODEL_LEN="${EVAL_MAX_MODEL_LEN:-8192}"

LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs}"
RUN_TAG="${RUN_TAG:-$(date '+%Y%m%d_%H%M%S')}"
RUN_LOG="${RUN_LOG:-${LOG_DIR}/qwen-large-tp4-then-eval-${RUN_TAG}.log}"
QWEN_SERVER_LOG="${QWEN_SERVER_LOG:-${LOG_DIR}/qwen-large-tp4-${RUN_TAG}.server.log}"
EVAL_SERVER_LOG="${EVAL_SERVER_LOG:-${LOG_DIR}/gpt-oss-tp4-${RUN_TAG}.server.log}"
PROGRESS_LOG="${PROGRESS_LOG:-${LOG_DIR}/qwen-large-progress-${RUN_TAG}.log}"

mkdir -p "${LOG_DIR}"
: >>"${RUN_LOG}"
exec >>"${RUN_LOG}" 2>&1

QWEN_SERVER_PID=""
EVAL_SERVER_PID=""
PROGRESS_MONITOR_PID=""

status() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

cleanup() {
    if [[ -n "${PROGRESS_MONITOR_PID}" ]]; then
        kill "${PROGRESS_MONITOR_PID}" 2>/dev/null || true
        wait "${PROGRESS_MONITOR_PID}" 2>/dev/null || true
    fi
    if [[ -n "${QWEN_SERVER_PID}" ]]; then
        kill "${QWEN_SERVER_PID}" 2>/dev/null || true
        wait "${QWEN_SERVER_PID}" 2>/dev/null || true
    fi
    if [[ -n "${EVAL_SERVER_PID}" ]]; then
        kill "${EVAL_SERVER_PID}" 2>/dev/null || true
        wait "${EVAL_SERVER_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

wait_for_health() {
    local port="$1"
    local pid="$2"
    local log_file="$3"
    local label="$4"
    local waited=0
    while (( waited < 1800 )); do
        if curl -fsS "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
            status "${label} ready on port=${port}"
            return 0
        fi
        if ! kill -0 "${pid}" 2>/dev/null; then
            status "${label} exited during startup"
            tail -n 120 "${log_file}" || true
            return 1
        fi
        sleep 5
        waited=$((waited + 5))
    done
    status "${label} timed out during startup"
    tail -n 120 "${log_file}" || true
    return 1
}

start_qwen_server() {
    status "Starting qwen_large TP4 vLLM on GPUs 0,1,2,3"
    CUDA_VISIBLE_DEVICES="0,1,2,3" \
    OPENAI_API_KEY="${OPENAI_API_KEY_LOCAL}" \
    "${VLLM_PYTHON_BIN}" -m vllm.entrypoints.openai.api_server \
        --model "${QWEN_MODEL_NAME}" \
        --served-model-name "${QWEN_SERVED_NAME}" \
        --port "${QWEN_PORT}" \
        --tensor-parallel-size "${QWEN_TP_SIZE}" \
        --gpu-memory-utilization "${QWEN_GPU_MEMORY_UTILIZATION}" \
        --max-model-len "${QWEN_MAX_MODEL_LEN}" \
        --trust-remote-code \
        >"${QWEN_SERVER_LOG}" 2>&1 &
    QWEN_SERVER_PID=$!
    wait_for_health "${QWEN_PORT}" "${QWEN_SERVER_PID}" "${QWEN_SERVER_LOG}" "qwen_large TP4 vLLM"
}

family_output_complete() {
    local family="$1"
    "${PYTHON_BIN}" - <<PY
import json
from pathlib import Path
import sys

root = Path("/data/minseo/KCC2026/korean-rubric-grounding-main")
sys.path.insert(0, str(root))
from src.utils import get_condition_specs, get_meta_eval_pairs_path, load_config  # noqa: E402

config = load_config(root / "configs/main.yaml")
pairs = [json.loads(line) for line in get_meta_eval_pairs_path().open(encoding="utf-8") if line.strip()]
pair_count = len(pairs)
prompt_count = len({row["prompt_id"] for row in pairs})

family = ${family@Q}
for spec in get_condition_specs(config):
    if spec.generator_family != family:
        continue
    path = root / "data/processed" / f"rubrics_{spec.method}__{spec.generator_family}.jsonl"
    if not path.exists():
        raise SystemExit(1)
    row_count = sum(1 for line in path.open(encoding="utf-8") if line.strip())
    expected = pair_count if spec.rubric_scope == "pair" else prompt_count
    if row_count != expected:
        raise SystemExit(1)
print("OK")
PY
}

start_progress_monitor() {
    (
        while true; do
            printf '\n[%s]\n' "$(date '+%Y-%m-%d %H:%M:%S')"
            bash "${REPO_ROOT}/scripts/show_rubric_progress.sh" || true
            sleep 60
        done
    ) >"${PROGRESS_LOG}" 2>&1 &
    PROGRESS_MONITOR_PID=$!
}

run_qwen_generation() {
    status "Starting qwen_large rubric generation with concurrency=${QWEN_CONCURRENCY}"
    OPENAI_BASE_URL="http://127.0.0.1:${QWEN_PORT}/v1" \
    OPENAI_API_KEY="${OPENAI_API_KEY_LOCAL}" \
    KCC_EMBED_DEVICE="cpu" \
    "${PYTHON_BIN}" -m src.generate_rubrics --config configs/main.yaml --generator_family qwen_large --concurrency "${QWEN_CONCURRENCY}"
}

start_eval_server() {
    status "Starting gpt-oss-120b TP4 vLLM on GPUs 0,1,2,3"
    CUDA_VISIBLE_DEVICES="0,1,2,3" \
    OPENAI_API_KEY="${OPENAI_API_KEY_LOCAL}" \
    "${EVAL_VLLM_PYTHON_BIN}" -m vllm.entrypoints.openai.api_server \
        --model "${EVAL_MODEL_NAME}" \
        --served-model-name "${EVAL_SERVED_NAME}" \
        --port "${EVAL_PORT}" \
        --tensor-parallel-size "${EVAL_TP_SIZE}" \
        --gpu-memory-utilization "${EVAL_GPU_MEMORY_UTILIZATION}" \
        --max-model-len "${EVAL_MAX_MODEL_LEN}" \
        --trust-remote-code \
        >"${EVAL_SERVER_LOG}" 2>&1 &
    EVAL_SERVER_PID=$!
    wait_for_health "${EVAL_PORT}" "${EVAL_SERVER_PID}" "${EVAL_SERVER_LOG}" "gpt-oss-120b TP4 vLLM"
}

run_eval() {
    status "Running pairwise evaluation"
    OPENAI_BASE_URL="http://127.0.0.1:${EVAL_PORT}/v1" \
    OPENAI_API_KEY="${OPENAI_API_KEY_LOCAL}" \
    "${PYTHON_BIN}" -m src.evaluate_pairs --config configs/main.yaml
    OPENAI_BASE_URL="http://127.0.0.1:${EVAL_PORT}/v1" \
    OPENAI_API_KEY="${OPENAI_API_KEY_LOCAL}" \
    "${PYTHON_BIN}" -m src.export_run_manifest --config configs/main.yaml
    status "Evaluation completed"
}

status "Preparing qwen_large TP4 resume + gpt-oss TP4 evaluation"
status "Logs: run=${RUN_LOG} qwen_server=${QWEN_SERVER_LOG} eval_server=${EVAL_SERVER_LOG} progress=${PROGRESS_LOG}"

pkill -f 'src.generate_rubrics --config configs/main.yaml --generator_family qwen_large' 2>/dev/null || true
pkill -f 'vllm.entrypoints.openai.api_server.*8102' 2>/dev/null || true
pkill -f 'vllm.entrypoints.openai.api_server.*8106' 2>/dev/null || true
pkill -f 'vllm.entrypoints.openai.api_server.*8012' 2>/dev/null || true

start_qwen_server
start_progress_monitor
if run_qwen_generation; then
    status "qwen_large generation finished cleanly"
else
    if family_output_complete "qwen_large" >/dev/null 2>&1; then
        status "qwen_large exited non-zero but outputs are complete; continuing"
    else
        status "qwen_large generation failed before completion"
        exit 1
    fi
fi

kill "${PROGRESS_MONITOR_PID}" 2>/dev/null || true
wait "${PROGRESS_MONITOR_PID}" 2>/dev/null || true
PROGRESS_MONITOR_PID=""

kill "${QWEN_SERVER_PID}" 2>/dev/null || true
wait "${QWEN_SERVER_PID}" 2>/dev/null || true
QWEN_SERVER_PID=""

start_eval_server
run_eval

status "qwen_large TP4 + gpt-oss evaluation flow complete"
