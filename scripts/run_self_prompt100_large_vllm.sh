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
CONFIG_PATH="${CONFIG_PATH:-configs/main.yaml}"
PROMPT_ID_FILE="${PROMPT_ID_FILE:-data/processed/rrd_sample_prompt_subset_100.ids.txt}"
OUTPUT_SPLIT="${OUTPUT_SPLIT:-prompt100}"
OPENAI_API_KEY_LOCAL="${OPENAI_API_KEY_LOCAL:-EMPTY}"
RESET_OUTPUTS="${RESET_OUTPUTS:-0}"
NO_FALLBACK="${NO_FALLBACK:-0}"
MAX_ROW_ATTEMPTS="${MAX_ROW_ATTEMPTS:-6}"

QWEN_VLLM_PYTHON_BIN="${QWEN_VLLM_PYTHON_BIN:-/tmp/venvs/rrd-qwen-vllm/bin/python}"
QWEN_MODEL_NAME="${QWEN_MODEL_NAME:-Qwen/Qwen3-32B}"
QWEN_SERVED_NAME="${QWEN_SERVED_NAME:-Qwen/Qwen3-32B}"
QWEN_PORT="${QWEN_PORT:-8110}"
QWEN_CUDA_DEVICES="${QWEN_CUDA_DEVICES:-0,1}"
QWEN_TP_SIZE="${QWEN_TP_SIZE:-2}"
QWEN_CONCURRENCY="${QWEN_CONCURRENCY:-8}"
QWEN_GPU_MEMORY_UTILIZATION="${QWEN_GPU_MEMORY_UTILIZATION:-0.90}"
QWEN_MAX_MODEL_LEN="${QWEN_MAX_MODEL_LEN:-32768}"

GEMMA_VLLM_PYTHON_BIN="${GEMMA_VLLM_PYTHON_BIN:-/tmp/venvs/vllm-gemma4-019/bin/python}"
GEMMA_MODEL_NAME="${GEMMA_MODEL_NAME:-google/gemma-4-31B-it}"
GEMMA_SERVED_NAME="${GEMMA_SERVED_NAME:-google/gemma-4-31B-it}"
GEMMA_PORT="${GEMMA_PORT:-8112}"
GEMMA_CUDA_DEVICES="${GEMMA_CUDA_DEVICES:-2,3}"
GEMMA_TP_SIZE="${GEMMA_TP_SIZE:-2}"
GEMMA_CONCURRENCY="${GEMMA_CONCURRENCY:-8}"
GEMMA_GPU_MEMORY_UTILIZATION="${GEMMA_GPU_MEMORY_UTILIZATION:-0.90}"
GEMMA_MAX_MODEL_LEN="${GEMMA_MAX_MODEL_LEN:-32768}"

LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs}"
RUN_TAG="${RUN_TAG:-$(date '+%Y%m%d_%H%M%S')}"
LAUNCH_LOG="${LAUNCH_LOG:-${LOG_DIR}/self-prompt100-large-${RUN_TAG}.log}"
PROGRESS_LOG="${PROGRESS_LOG:-${LOG_DIR}/self-prompt100-large-${RUN_TAG}.progress.log}"

mkdir -p "${LOG_DIR}"
: >>"${LAUNCH_LOG}"
exec > >(tee -a "${LAUNCH_LOG}") 2>&1

declare -A SERVER_PIDS
declare -A WORKER_PIDS
PROGRESS_MONITOR_PID=""
FAILURES=0

status() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

cleanup() {
    if [[ -n "${PROGRESS_MONITOR_PID}" ]]; then
        kill "${PROGRESS_MONITOR_PID}" 2>/dev/null || true
        wait "${PROGRESS_MONITOR_PID}" 2>/dev/null || true
        PROGRESS_MONITOR_PID=""
    fi
    for pid in "${WORKER_PIDS[@]:-}"; do
        kill "${pid}" 2>/dev/null || true
        wait "${pid}" 2>/dev/null || true
    done
    for pid in "${SERVER_PIDS[@]:-}"; do
        kill "${pid}" 2>/dev/null || true
        wait "${pid}" 2>/dev/null || true
    done
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
    local method="$2"
    printf '%s/data/processed/rubrics_%s__%s__%s.jsonl' "${REPO_ROOT}" "${method}" "${family}" "${OUTPUT_SPLIT}"
}

subset_progress_path() {
    local family="$1"
    local method="$2"
    printf '%s.progress.json' "$(subset_output_path "${family}" "${method}")"
}

remove_subset_outputs() {
    local family="$1"
    local method="$2"
    rm -f \
        "$(subset_output_path "${family}" "${method}")" \
        "$(subset_progress_path "${family}" "${method}")"
}

verify_subset_output() {
    local family="$1"
    local method="$2"
    "${PYTHON_BIN}" - <<PY
from pathlib import Path

path = Path(${REPO_ROOT@Q}) / "data/processed" / f"rubrics_${method}__${family}__${OUTPUT_SPLIT}.jsonl"
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
    local tp_size="$7"
    local gpu_memory_utilization="$8"
    local max_model_len="$9"
    local server_log="${LOG_DIR}/${family}-self-prompt100-${RUN_TAG}.server.log"

    pkill -f "vllm.entrypoints.openai.api_server.*${port}" 2>/dev/null || true

    status "Starting ${family} server model=${model_name} cuda=${cuda_devices} tp=${tp_size} port=${port}"
    CUDA_VISIBLE_DEVICES="${cuda_devices}" \
    OPENAI_API_KEY="${OPENAI_API_KEY_LOCAL}" \
    "${python_bin}" -m vllm.entrypoints.openai.api_server \
        --model "${model_name}" \
        --served-model-name "${served_name}" \
        --port "${port}" \
        --tensor-parallel-size "${tp_size}" \
        --gpu-memory-utilization "${gpu_memory_utilization}" \
        --max-model-len "${max_model_len}" \
        --trust-remote-code \
        >"${server_log}" 2>&1 &
    SERVER_PIDS["${family}"]=$!
    wait_for_health "${port}" "${SERVER_PIDS[${family}]}" "${server_log}" "${family} vLLM"
}

run_generation() {
    local family="$1"
    local method="$2"
    local port="$3"
    local concurrency="$4"
    local generation_log="${LOG_DIR}/${family}-${method}-${RUN_TAG}.generate.log"

    if [[ "${RESET_OUTPUTS}" == "1" ]]; then
        status "Resetting subset outputs family=${family} method=${method}"
        remove_subset_outputs "${family}" "${method}"
    fi

    local -a cmd=(
        "${PYTHON_BIN}" -m src.generate_rubrics
        --config "${CONFIG_PATH}"
        --generator_family "${family}"
        --method "${method}"
        --concurrency "${concurrency}"
        --prompt-id-file "${PROMPT_ID_FILE}"
        --output-split "${OUTPUT_SPLIT}"
        --max-row-attempts "${MAX_ROW_ATTEMPTS}"
    )
    if [[ "${NO_FALLBACK}" == "1" ]]; then
        cmd+=(--no-fallback)
    fi

    status "Generating family=${family} method=${method} port=${port} concurrency=${concurrency}"
    OPENAI_BASE_URL="http://127.0.0.1:${port}/v1" \
    OPENAI_API_KEY="${OPENAI_API_KEY_LOCAL}" \
    KCC_RUBRIC_GENERATION_BACKEND="openai_compatible" \
    KCC_EMBED_DEVICE="cpu" \
    "${cmd[@]}" >"${generation_log}" 2>&1
    verify_subset_output "${family}" "${method}"
}

run_family_lane() {
    local family="$1"
    local port="$2"
    local concurrency="$3"

    run_generation "${family}" "rar_pairwise_self" "${port}" "${concurrency}"
    run_generation "${family}" "rrd_pairwise_self" "${port}" "${concurrency}"
    status "Completed family lane ${family}"
}

status "Preparing prompt-only self rubric generation for 100 prompts"
status "qwen_large -> GPUs ${QWEN_CUDA_DEVICES}; gemma_large -> GPUs ${GEMMA_CUDA_DEVICES}"
status "Config=${CONFIG_PATH} prompt_ids=${PROMPT_ID_FILE} output_split=${OUTPUT_SPLIT}"
status "Logs: launch=${LAUNCH_LOG} progress=${PROGRESS_LOG}"

verify_prompt_subset
start_server "qwen_large" "${QWEN_MODEL_NAME}" "${QWEN_SERVED_NAME}" "${QWEN_PORT}" "${QWEN_VLLM_PYTHON_BIN}" "${QWEN_CUDA_DEVICES}" "${QWEN_TP_SIZE}" "${QWEN_GPU_MEMORY_UTILIZATION}" "${QWEN_MAX_MODEL_LEN}"
start_server "gemma_large" "${GEMMA_MODEL_NAME}" "${GEMMA_SERVED_NAME}" "${GEMMA_PORT}" "${GEMMA_VLLM_PYTHON_BIN}" "${GEMMA_CUDA_DEVICES}" "${GEMMA_TP_SIZE}" "${GEMMA_GPU_MEMORY_UTILIZATION}" "${GEMMA_MAX_MODEL_LEN}"
start_progress_monitor

run_family_lane "qwen_large" "${QWEN_PORT}" "${QWEN_CONCURRENCY}" &
WORKER_PIDS["qwen_large"]=$!

run_family_lane "gemma_large" "${GEMMA_PORT}" "${GEMMA_CONCURRENCY}" &
WORKER_PIDS["gemma_large"]=$!

for family in "qwen_large" "gemma_large"; do
    pid="${WORKER_PIDS[${family}]}"
    if wait "${pid}"; then
        status "Lane completed family=${family}"
    else
        status "Lane failed family=${family}"
        FAILURES=1
    fi
done

if [[ "${FAILURES}" != "0" ]]; then
    status "One or more lanes failed"
    exit 1
fi

status "Completed prompt-only self rubric generation for qwen_large and gemma_large"
