#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || command -v python)}"
VLLM_PYTHON_BIN="${VLLM_PYTHON_BIN:-${PYTHON_BIN}}"
CONFIG_PATH="${CONFIG_PATH:-configs/qwen_small_tp4_rrd.yaml}"
OPENAI_API_KEY_LOCAL="${OPENAI_API_KEY_LOCAL:-EMPTY}"

QWEN_MODEL_NAME="${QWEN_MODEL_NAME:-Qwen/Qwen3-4B-FP8}"
QWEN_SERVED_NAME="${QWEN_SERVED_NAME:-Qwen/Qwen3-4B-FP8}"
QWEN_PORT="${QWEN_PORT:-8101}"
QWEN_TP_SIZE="${QWEN_TP_SIZE:-4}"
QWEN_GPU_MEMORY_UTILIZATION="${QWEN_GPU_MEMORY_UTILIZATION:-0.92}"
QWEN_MAX_MODEL_LEN="${QWEN_MAX_MODEL_LEN:-32768}"
QWEN_CONCURRENCY="${QWEN_CONCURRENCY:-30}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3}"

LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs}"
RUN_TAG="${RUN_TAG:-$(date '+%Y%m%d_%H%M%S')}"
RUN_LOG="${RUN_LOG:-${LOG_DIR}/qwen-small-tp4-rrd-${RUN_TAG}.log}"
SERVER_LOG="${SERVER_LOG:-${LOG_DIR}/qwen-small-tp4-rrd-${RUN_TAG}.server.log}"
PROGRESS_LOG="${PROGRESS_LOG:-${LOG_DIR}/qwen-small-tp4-rrd-${RUN_TAG}.progress.log}"

mkdir -p "${LOG_DIR}"
: >>"${RUN_LOG}"
exec >>"${RUN_LOG}" 2>&1

SERVER_PID=""
PROGRESS_MONITOR_PID=""

status() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

cleanup() {
    if [[ -n "${PROGRESS_MONITOR_PID}" ]]; then
        kill "${PROGRESS_MONITOR_PID}" 2>/dev/null || true
        wait "${PROGRESS_MONITOR_PID}" 2>/dev/null || true
        PROGRESS_MONITOR_PID=""
    fi
    if [[ -n "${SERVER_PID}" ]]; then
        kill "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
        SERVER_PID=""
    fi
}
trap cleanup EXIT INT TERM

wait_for_health() {
    local port="$1"
    local pid="$2"
    local log_file="$3"
    local waited=0
    while (( waited < 1800 )); do
        if curl -fsS "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
            status "vLLM ready on port=${port}"
            return 0
        fi
        if ! kill -0 "${pid}" 2>/dev/null; then
            status "vLLM exited during startup"
            tail -n 120 "${log_file}" || true
            return 1
        fi
        sleep 5
        waited=$((waited + 5))
    done
    status "Timed out waiting for vLLM health endpoint"
    tail -n 120 "${log_file}" || true
    return 1
}

remove_old_rrd_outputs() {
    rm -f \
        data/processed/rubrics_rrd_pairwise_reference__qwen_small.jsonl \
        data/processed/rubrics_rrd_pairwise_reference__qwen_small.jsonl.progress.json \
        data/processed/rubrics_rrd_pairwise_sample__qwen_small.jsonl \
        data/processed/rubrics_rrd_pairwise_sample__qwen_small.jsonl.progress.json
}

start_progress_monitor() {
    (
        while true; do
            printf '\n[%s]\n' "$(date '+%Y-%m-%d %H:%M:%S')"
            PYTHON_BIN="${PYTHON_BIN}" bash "${REPO_ROOT}/scripts/show_rubric_progress.sh" || true
            sleep 60
        done
    ) >"${PROGRESS_LOG}" 2>&1 &
    PROGRESS_MONITOR_PID=$!
}

start_server() {
    status "Starting qwen_small TP4 vLLM on GPUs ${CUDA_DEVICES}"
    CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" \
    OPENAI_API_KEY="${OPENAI_API_KEY_LOCAL}" \
    "${VLLM_PYTHON_BIN}" -m vllm.entrypoints.openai.api_server \
        --model "${QWEN_MODEL_NAME}" \
        --served-model-name "${QWEN_SERVED_NAME}" \
        --port "${QWEN_PORT}" \
        --tensor-parallel-size "${QWEN_TP_SIZE}" \
        --gpu-memory-utilization "${QWEN_GPU_MEMORY_UTILIZATION}" \
        --max-model-len "${QWEN_MAX_MODEL_LEN}" \
        --trust-remote-code \
        >"${SERVER_LOG}" 2>&1 &
    SERVER_PID=$!
    wait_for_health "${QWEN_PORT}" "${SERVER_PID}" "${SERVER_LOG}"
}

run_generation() {
    local method="$1"
    status "Generating ${method} with qwen_small TP4"
    OPENAI_BASE_URL="http://127.0.0.1:${QWEN_PORT}/v1" \
    OPENAI_API_KEY="${OPENAI_API_KEY_LOCAL}" \
    KCC_RUBRIC_GENERATION_BACKEND="openai_compatible" \
    KCC_EMBED_DEVICE="cpu" \
    "${PYTHON_BIN}" -m src.generate_rubrics \
        --config "${CONFIG_PATH}" \
        --generator_family qwen_small \
        --method "${method}" \
        --concurrency "${QWEN_CONCURRENCY}"
}

status "Preparing qwen_small TP4 RRD generation"
status "Logs: run=${RUN_LOG} server=${SERVER_LOG} progress=${PROGRESS_LOG}"
pkill -f "vllm.entrypoints.openai.api_server.*${QWEN_PORT}" 2>/dev/null || true
pkill -f "src.generate_rubrics --config ${CONFIG_PATH} --generator_family qwen_small" 2>/dev/null || true
remove_old_rrd_outputs
start_server
start_progress_monitor
run_generation rrd_pairwise_reference
run_generation rrd_pairwise_sample
status "qwen_small TP4 RRD generation finished"
