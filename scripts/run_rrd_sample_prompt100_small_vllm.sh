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
CONFIG_PATH="${CONFIG_PATH:-configs/rrd_sample_prompt100_small.yaml}"
PROMPT_ID_FILE="${PROMPT_ID_FILE:-data/processed/rrd_sample_prompt_subset_100.ids.txt}"
OUTPUT_SPLIT="${OUTPUT_SPLIT:-prompt100}"
METHOD="${METHOD:-rrd_pairwise_sample}"
CONCURRENCY="${CONCURRENCY:-4}"
OPENAI_API_KEY_LOCAL="${OPENAI_API_KEY_LOCAL:-EMPTY}"
RESET_OUTPUTS="${RESET_OUTPUTS:-0}"

QWEN_VLLM_PYTHON_BIN="${QWEN_VLLM_PYTHON_BIN:-/tmp/venvs/rrd-qwen-vllm/bin/python}"
QWEN_MODEL_NAME="${QWEN_MODEL_NAME:-Qwen/Qwen3-4B-FP8}"
QWEN_SERVED_NAME="${QWEN_SERVED_NAME:-Qwen/Qwen3-4B-FP8}"
QWEN_PORT="${QWEN_PORT:-8101}"
QWEN_CUDA_DEVICES="${QWEN_CUDA_DEVICES:-0}"
QWEN_GPU_MEMORY_UTILIZATION="${QWEN_GPU_MEMORY_UTILIZATION:-0.92}"
QWEN_MAX_MODEL_LEN="${QWEN_MAX_MODEL_LEN:-32768}"

GEMMA_VLLM_PYTHON_BIN="${GEMMA_VLLM_PYTHON_BIN:-/tmp/venvs/vllm-gemma4-019/bin/python}"
GEMMA_MODEL_NAME="${GEMMA_MODEL_NAME:-google/gemma-4-E4B-it}"
GEMMA_SERVED_NAME="${GEMMA_SERVED_NAME:-google/gemma-4-E4B-it}"
GEMMA_PORT="${GEMMA_PORT:-8103}"
GEMMA_CUDA_DEVICES="${GEMMA_CUDA_DEVICES:-0}"
GEMMA_GPU_MEMORY_UTILIZATION="${GEMMA_GPU_MEMORY_UTILIZATION:-0.92}"
GEMMA_MAX_MODEL_LEN="${GEMMA_MAX_MODEL_LEN:-32768}"

LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs}"
RUN_TAG="${RUN_TAG:-$(date '+%Y%m%d_%H%M%S')}"
LAUNCH_LOG="${LAUNCH_LOG:-${LOG_DIR}/rrd-sample-prompt100-small-${RUN_TAG}.log}"
PROGRESS_LOG="${PROGRESS_LOG:-${LOG_DIR}/rrd-sample-prompt100-small-${RUN_TAG}.progress.log}"

mkdir -p "${LOG_DIR}"
: >>"${LAUNCH_LOG}"
exec >>"${LAUNCH_LOG}" 2>&1

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
    status "Timed out waiting for vLLM health endpoint on port=${port}"
    tail -n 120 "${log_file}" || true
    return 1
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

subset_output_path() {
    local family="$1"
    printf '%s/data/processed/rubrics_%s__%s__%s.jsonl' "${REPO_ROOT}" "${METHOD}" "${family}" "${OUTPUT_SPLIT}"
}

subset_progress_path() {
    local family="$1"
    printf '%s.progress.json' "$(subset_output_path "${family}")"
}

remove_subset_outputs() {
    local family="$1"
    rm -f \
        "$(subset_output_path "${family}")" \
        "$(subset_progress_path "${family}")"
}

verify_subset_output() {
    local family="$1"
    "${PYTHON_BIN}" - <<PY
from pathlib import Path

path = Path(${REPO_ROOT@Q}) / "data/processed" / f"rubrics_${METHOD}__${family}__${OUTPUT_SPLIT}.jsonl"
if not path.exists():
    raise SystemExit(f"Missing output file: {path}")
row_count = sum(1 for line in path.open(encoding="utf-8") if line.strip())
if row_count != 100:
    raise SystemExit(f"Expected 100 rows in {path}, found {row_count}")
print(f"Verified {path.name}: {row_count} rows")
PY
}

start_server() {
    local family="$1"
    local model_name="$2"
    local served_name="$3"
    local port="$4"
    local python_bin="$5"
    local cuda_devices="$6"
    local gpu_memory_utilization="$7"
    local max_model_len="$8"
    local server_log="${LOG_DIR}/${family}-prompt100-${RUN_TAG}.server.log"

    pkill -f "vllm.entrypoints.openai.api_server.*${port}" 2>/dev/null || true

    status "Starting ${family} server model=${model_name} cuda=${cuda_devices} port=${port}"
    CUDA_VISIBLE_DEVICES="${cuda_devices}" \
    OPENAI_API_KEY="${OPENAI_API_KEY_LOCAL}" \
    "${python_bin}" -m vllm.entrypoints.openai.api_server \
        --model "${model_name}" \
        --served-model-name "${served_name}" \
        --port "${port}" \
        --gpu-memory-utilization "${gpu_memory_utilization}" \
        --max-model-len "${max_model_len}" \
        --tensor-parallel-size 1 \
        --trust-remote-code \
        >"${server_log}" 2>&1 &
    SERVER_PID=$!
    wait_for_health "${port}" "${SERVER_PID}" "${server_log}"
}

stop_server() {
    if [[ -n "${SERVER_PID}" ]]; then
        status "Stopping server pid=${SERVER_PID}"
        kill "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
        SERVER_PID=""
    fi
}

run_generation() {
    local family="$1"
    local port="$2"
    local generation_log="${LOG_DIR}/${family}-prompt100-${RUN_TAG}.generate.log"

    if [[ "${RESET_OUTPUTS}" == "1" ]]; then
        status "Resetting subset outputs for ${family}"
        remove_subset_outputs "${family}"
    fi

    status "Generating ${METHOD} for ${family} using prompt subset=${PROMPT_ID_FILE}"
    OPENAI_BASE_URL="http://127.0.0.1:${port}/v1" \
    OPENAI_API_KEY="${OPENAI_API_KEY_LOCAL}" \
    KCC_RUBRIC_GENERATION_BACKEND="openai_compatible" \
    KCC_EMBED_DEVICE="cpu" \
    "${PYTHON_BIN}" -m src.generate_rubrics \
        --config "${CONFIG_PATH}" \
        --generator_family "${family}" \
        --method "${METHOD}" \
        --concurrency "${CONCURRENCY}" \
        --prompt-id-file "${PROMPT_ID_FILE}" \
        --output-split "${OUTPUT_SPLIT}" \
        >"${generation_log}" 2>&1
}

run_family() {
    local family="$1"
    case "${family}" in
        qwen_small)
            start_server "${family}" "${QWEN_MODEL_NAME}" "${QWEN_SERVED_NAME}" "${QWEN_PORT}" "${QWEN_VLLM_PYTHON_BIN}" "${QWEN_CUDA_DEVICES}" "${QWEN_GPU_MEMORY_UTILIZATION}" "${QWEN_MAX_MODEL_LEN}"
            run_generation "${family}" "${QWEN_PORT}"
            verify_subset_output "${family}"
            stop_server
            ;;
        gemma_small)
            start_server "${family}" "${GEMMA_MODEL_NAME}" "${GEMMA_SERVED_NAME}" "${GEMMA_PORT}" "${GEMMA_VLLM_PYTHON_BIN}" "${GEMMA_CUDA_DEVICES}" "${GEMMA_GPU_MEMORY_UTILIZATION}" "${GEMMA_MAX_MODEL_LEN}"
            run_generation "${family}" "${GEMMA_PORT}"
            verify_subset_output "${family}"
            stop_server
            ;;
        *)
            echo "Unsupported family: ${family}" >&2
            exit 1
            ;;
    esac
}

status "Preparing 100-prompt RRD-sample run for qwen_small and gemma_small"
status "Config=${CONFIG_PATH} prompt_ids=${PROMPT_ID_FILE} output_split=${OUTPUT_SPLIT}"
status "Logs: launch=${LAUNCH_LOG} progress=${PROGRESS_LOG}"
verify_prompt_subset
start_progress_monitor
run_family qwen_small
run_family gemma_small
status "Completed 100-prompt RRD-sample generation for qwen_small and gemma_small"
