#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

DEFAULT_PYTHON_BIN="$(command -v python3 || command -v python)"
PYTHON_BIN="${PYTHON_BIN:-${DEFAULT_PYTHON_BIN}}"

CONFIG_PATH="${CONFIG_PATH:-configs/rrd_sample_prompt100_gemma_e4b.yaml}"
PROMPT_ID_FILE="${PROMPT_ID_FILE:-data/processed/rrd_sample_prompt_subset_100.ids.txt}"
OUTPUT_SPLIT="${OUTPUT_SPLIT:-prompt100}"
METHOD="${METHOD:-rrd_pairwise_sample}"
GENERATOR_FAMILY="${GENERATOR_FAMILY:-gemma_small}"
CONCURRENCY="${CONCURRENCY:-30}"
OPENAI_API_KEY_LOCAL="${OPENAI_API_KEY_LOCAL:-EMPTY}"
RESET_OUTPUTS="${RESET_OUTPUTS:-0}"

DEFAULT_GEMMA_VLLM_PYTHON_BIN="/data/minseo/envs/vllm-gemma4-019/bin/python"
if [[ ! -x "${DEFAULT_GEMMA_VLLM_PYTHON_BIN}" ]]; then
    DEFAULT_GEMMA_VLLM_PYTHON_BIN="${DEFAULT_PYTHON_BIN}"
fi

GEMMA_VLLM_PYTHON_BIN="${GEMMA_VLLM_PYTHON_BIN:-${DEFAULT_GEMMA_VLLM_PYTHON_BIN}}"
GEMMA_MODEL_NAME="${GEMMA_MODEL_NAME:-google/gemma-4-E4B-it}"
GEMMA_SERVED_NAME="${GEMMA_SERVED_NAME:-google/gemma-4-E4B-it}"
GEMMA_PORT="${GEMMA_PORT:-8103}"
GEMMA_CUDA_DEVICES="${GEMMA_CUDA_DEVICES:-2}"
GEMMA_GPU_MEMORY_UTILIZATION="${GEMMA_GPU_MEMORY_UTILIZATION:-0.92}"
GEMMA_MAX_MODEL_LEN="${GEMMA_MAX_MODEL_LEN:-32768}"
GEMMA_TENSOR_PARALLEL_SIZE="${GEMMA_TENSOR_PARALLEL_SIZE:-1}"
GEMMA_MAX_NUM_SEQS="${GEMMA_MAX_NUM_SEQS:-4}"
GEMMA_ENFORCE_EAGER="${GEMMA_ENFORCE_EAGER:-0}"

LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs}"
RUN_TAG="${RUN_TAG:-$(date '+%Y%m%d_%H%M%S')}"
LAUNCH_LOG="${LAUNCH_LOG:-${LOG_DIR}/${METHOD}-${OUTPUT_SPLIT}-gemma-e4b-${RUN_TAG}.log}"
PROGRESS_LOG="${PROGRESS_LOG:-${LOG_DIR}/${METHOD}-${OUTPUT_SPLIT}-gemma-e4b-${RUN_TAG}.progress.log}"

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

subset_expected_rows() {
    "${PYTHON_BIN}" - <<PY
from pathlib import Path
import json

prompt_ids = {
    line.strip()
    for line in Path(${PROMPT_ID_FILE@Q}).read_text(encoding="utf-8").splitlines()
    if line.strip()
}
if ${METHOD@Q} in {"rar_pairwise_reference", "rrd_pairwise_reference"}:
    path = Path(${REPO_ROOT@Q}) / "data/processed/meta_eval_pairs.jsonl"
    count = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["prompt_id"] in prompt_ids:
                count += 1
    print(count)
else:
    print(len(prompt_ids))
PY
}

wait_for_api_ready() {
    local port="$1"
    local pid="$2"
    local log_file="$3"
    local served_name="$4"
    local waited=0
    while (( waited < 1800 )); do
        if curl -fsS "http://127.0.0.1:${port}/v1/models" | grep -F "\"${served_name}\"" >/dev/null 2>&1; then
            local completion_code=""
            completion_code="$(
                curl -sS -o /tmp/kcc_gemma_preflight_"${port}".json -w '%{http_code}' \
                    "http://127.0.0.1:${port}/v1/chat/completions" \
                    -H 'Content-Type: application/json' \
                    -H "Authorization: Bearer ${OPENAI_API_KEY_LOCAL}" \
                    -d "$(cat <<JSON
{"model":"${served_name}","messages":[{"role":"user","content":"ping"}],"max_tokens":8,"temperature":0}
JSON
)" || true
            )"
            if [[ "${completion_code}" == "200" ]]; then
                status "vLLM ready on port=${port} model=${served_name}"
                return 0
            fi
        fi
        if ! kill -0 "${pid}" 2>/dev/null; then
            status "vLLM exited during startup"
            tail -n 120 "${log_file}" || true
            return 1
        fi
        sleep 5
        waited=$((waited + 5))
    done
    status "Timed out waiting for vLLM API readiness on port=${port}"
    tail -n 120 "${log_file}" || true
    return 1
}

subset_output_path() {
    printf '%s/data/processed/rubrics_%s__%s__%s.jsonl' "${REPO_ROOT}" "${METHOD}" "${GENERATOR_FAMILY}" "${OUTPUT_SPLIT}"
}

subset_progress_path() {
    printf '%s.progress.json' "$(subset_output_path)"
}

remove_subset_outputs() {
    rm -f \
        "$(subset_output_path)" \
        "$(subset_progress_path)"
}

verify_subset_output() {
    local expected_rows=""
    expected_rows="$(subset_expected_rows)"
    "${PYTHON_BIN}" - <<PY
from pathlib import Path

path = Path(${REPO_ROOT@Q}) / "data/processed" / f"rubrics_${METHOD}__${GENERATOR_FAMILY}__${OUTPUT_SPLIT}.jsonl"
if not path.exists():
    raise SystemExit(f"Missing output file: {path}")
row_count = sum(1 for line in path.open(encoding="utf-8") if line.strip())
expected_rows = int(${expected_rows@Q})
if row_count != expected_rows:
    raise SystemExit(f"Expected {expected_rows} rows in {path}, found {row_count}")
print(f"Verified {path.name}: {row_count} rows")
PY
}

start_progress_monitor() {
    (
        while true; do
            printf '\n[%s]\n' "$(date '+%Y-%m-%d %H:%M:%S')"
            if [[ -f "$(subset_progress_path)" ]]; then
                "${PYTHON_BIN}" - <<PY
import json
from pathlib import Path

path = Path(${REPO_ROOT@Q}) / "data/processed" / f"rubrics_${METHOD}__${GENERATOR_FAMILY}__${OUTPUT_SPLIT}.jsonl.progress.json"
payload = json.loads(path.read_text(encoding="utf-8"))
done = int(payload.get("completed_rows", 0))
total = int(payload.get("expected_rows", 0))
remaining = int(payload.get("remaining_rows", max(total - done, 0)))
pct = (done / total * 100.0) if total else 0.0
print(f"{payload.get('generator_family')} {payload.get('method')} {done}/{total} {pct:.2f}% remaining={remaining}")
PY
            else
                echo "Waiting for progress file: $(subset_progress_path)"
            fi
            sleep 60
        done
    ) >"${PROGRESS_LOG}" 2>&1 &
    PROGRESS_MONITOR_PID=$!
}

start_server() {
    local server_log="${LOG_DIR}/${METHOD}-${OUTPUT_SPLIT}-gemma-e4b-${RUN_TAG}.server.log"
    local -a extra_server_args=()

    local old_pids=""
    old_pids="$(lsof -tiTCP:${GEMMA_PORT} -sTCP:LISTEN 2>/dev/null || true)"
    if [[ -n "${old_pids}" ]]; then
        status "Killing existing listeners on port=${GEMMA_PORT}: ${old_pids}"
        kill ${old_pids} 2>/dev/null || true
        sleep 2
    fi

    if [[ "${GEMMA_ENFORCE_EAGER}" == "1" ]]; then
        extra_server_args+=(--enforce-eager)
    fi

    status "Starting ${GENERATOR_FAMILY} server model=${GEMMA_MODEL_NAME} cuda=${GEMMA_CUDA_DEVICES} tp=${GEMMA_TENSOR_PARALLEL_SIZE} max_num_seqs=${GEMMA_MAX_NUM_SEQS} enforce_eager=${GEMMA_ENFORCE_EAGER} port=${GEMMA_PORT}"
    CUDA_VISIBLE_DEVICES="${GEMMA_CUDA_DEVICES}" \
    OPENAI_API_KEY="${OPENAI_API_KEY_LOCAL}" \
    "${GEMMA_VLLM_PYTHON_BIN}" -m vllm.entrypoints.openai.api_server \
        --model "${GEMMA_MODEL_NAME}" \
        --served-model-name "${GEMMA_SERVED_NAME}" \
        --port "${GEMMA_PORT}" \
        --gpu-memory-utilization "${GEMMA_GPU_MEMORY_UTILIZATION}" \
        --max-model-len "${GEMMA_MAX_MODEL_LEN}" \
        --max-num-seqs "${GEMMA_MAX_NUM_SEQS}" \
        --tensor-parallel-size "${GEMMA_TENSOR_PARALLEL_SIZE}" \
        --trust-remote-code \
        "${extra_server_args[@]}" \
        >"${server_log}" 2>&1 &
    SERVER_PID=$!
    wait_for_api_ready "${GEMMA_PORT}" "${SERVER_PID}" "${server_log}" "${GEMMA_SERVED_NAME}"
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
    local generation_log="${LOG_DIR}/${METHOD}-${OUTPUT_SPLIT}-gemma-e4b-${RUN_TAG}.generate.log"

    if [[ "${RESET_OUTPUTS}" == "1" ]]; then
        status "Resetting subset outputs for ${GENERATOR_FAMILY}"
        remove_subset_outputs
    fi

    status "Generating ${METHOD} for ${GENERATOR_FAMILY} using prompt subset=${PROMPT_ID_FILE}"
    OPENAI_BASE_URL="http://127.0.0.1:${GEMMA_PORT}/v1" \
    OPENAI_API_KEY="${OPENAI_API_KEY_LOCAL}" \
    KCC_RUBRIC_GENERATION_BACKEND="openai_compatible" \
    KCC_EMBED_DEVICE="cpu" \
    "${PYTHON_BIN}" -m src.generate_rubrics \
        --config "${CONFIG_PATH}" \
        --generator_family "${GENERATOR_FAMILY}" \
        --method "${METHOD}" \
        --concurrency "${CONCURRENCY}" \
        --prompt-id-file "${PROMPT_ID_FILE}" \
        --output-split "${OUTPUT_SPLIT}" \
        >"${generation_log}" 2>&1
}

status "Preparing 100-prompt ${METHOD} run for ${GENERATOR_FAMILY}"
status "Config=${CONFIG_PATH} prompt_ids=${PROMPT_ID_FILE} output_split=${OUTPUT_SPLIT}"
status "Gemma vLLM python=${GEMMA_VLLM_PYTHON_BIN}"
status "Logs: launch=${LAUNCH_LOG} progress=${PROGRESS_LOG}"
verify_prompt_subset
start_progress_monitor
start_server
run_generation
verify_subset_output
stop_server
status "Completed 100-prompt ${METHOD} generation for ${GENERATOR_FAMILY}"
