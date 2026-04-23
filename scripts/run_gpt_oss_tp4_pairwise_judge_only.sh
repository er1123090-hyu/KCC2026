#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
VLLM_PYTHON_BIN="${VLLM_PYTHON_BIN:-/tmp/venvs/rrd-qwen-vllm/bin/python}"
OPENAI_API_KEY_LOCAL="${OPENAI_API_KEY_LOCAL:-EMPTY}"
PORT="${PORT:-8012}"
MODEL_NAME="${MODEL_NAME:-openai/gpt-oss-120b}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-gpt-oss-120b}"
TP_SIZE="${TP_SIZE:-4}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.92}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"

LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs}"
RUN_TAG="${RUN_TAG:-$(date '+%Y%m%d_%H%M%S')}"
RUN_LOG="${RUN_LOG:-${LOG_DIR}/gpt-oss-pairwise-only-${RUN_TAG}.log}"
SERVER_LOG="${SERVER_LOG:-${LOG_DIR}/gpt-oss-pairwise-only-${RUN_TAG}.server.log}"
mkdir -p "${LOG_DIR}"
: >>"${RUN_LOG}"
exec >>"${RUN_LOG}" 2>&1

SERVER_PID=""

status() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

cleanup() {
    if [[ -n "${SERVER_PID}" ]]; then
        kill "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

wait_for_health() {
    local waited=0
    while (( waited < 1800 )); do
        if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
            status "gpt-oss TP4 ready"
            return 0
        fi
        if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
            status "gpt-oss TP4 exited during startup"
            tail -n 120 "${SERVER_LOG}" || true
            return 1
        fi
        sleep 5
        waited=$((waited + 5))
    done
    status "gpt-oss TP4 timed out during startup"
    tail -n 120 "${SERVER_LOG}" || true
    return 1
}

status "Starting gpt-oss pairwise-judge-only evaluation"
pkill -f 'vllm.entrypoints.openai.api_server.*8012' 2>/dev/null || true
pkill -f 'src.evaluate_pairwise_judges_only --config configs/main.yaml' 2>/dev/null || true

CUDA_VISIBLE_DEVICES="0,1,2,3" \
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

OPENAI_BASE_URL="http://127.0.0.1:${PORT}/v1" \
OPENAI_API_KEY="${OPENAI_API_KEY_LOCAL}" \
"${PYTHON_BIN}" -m src.evaluate_pairwise_judges_only --config configs/main.yaml

status "Pairwise-judge-only evaluation completed"
