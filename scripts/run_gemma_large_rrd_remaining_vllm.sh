#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

DEFAULT_PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
if [[ ! -x "${DEFAULT_PYTHON_BIN}" ]]; then
    DEFAULT_PYTHON_BIN="$(command -v python3 || command -v python)"
fi
PYTHON_BIN="${PYTHON_BIN:-${DEFAULT_PYTHON_BIN}}"

DEFAULT_GEMMA_VLLM_PYTHON_BIN="/data/minseo/envs/vllm-gemma4-019/bin/python"
if [[ ! -x "${DEFAULT_GEMMA_VLLM_PYTHON_BIN}" ]]; then
    DEFAULT_GEMMA_VLLM_PYTHON_BIN="/tmp/venvs/vllm-gemma4-019/bin/python"
fi
if [[ ! -x "${DEFAULT_GEMMA_VLLM_PYTHON_BIN}" ]]; then
    DEFAULT_GEMMA_VLLM_PYTHON_BIN="${DEFAULT_PYTHON_BIN}"
fi

GEMMA_VLLM_PYTHON_BIN="${GEMMA_VLLM_PYTHON_BIN:-${DEFAULT_GEMMA_VLLM_PYTHON_BIN}}"
CONFIG_PATH="${CONFIG_PATH:-configs/main.yaml}"
GENERATOR_FAMILY="${GENERATOR_FAMILY:-gemma_large}"
GEMMA_MODEL_NAME="${GEMMA_MODEL_NAME:-google/gemma-4-31B-it}"
GEMMA_SERVED_NAME="${GEMMA_SERVED_NAME:-google/gemma-4-31B-it}"
GEMMA_PORT="${GEMMA_PORT:-8104}"
GEMMA_CUDA_DEVICES="${GEMMA_CUDA_DEVICES:-0,1,2,3}"
GEMMA_TENSOR_PARALLEL_SIZE="${GEMMA_TENSOR_PARALLEL_SIZE:-4}"
GEMMA_GPU_MEMORY_UTILIZATION="${GEMMA_GPU_MEMORY_UTILIZATION:-0.92}"
GEMMA_MAX_MODEL_LEN="${GEMMA_MAX_MODEL_LEN:-32768}"
GEMMA_MAX_NUM_SEQS="${GEMMA_MAX_NUM_SEQS:-24}"
CONCURRENCY="${CONCURRENCY:-24}"
MAX_ROW_ATTEMPTS="${MAX_ROW_ATTEMPTS:-8}"
MAX_METHOD_ATTEMPTS="${MAX_METHOD_ATTEMPTS:-6}"
OPENAI_API_KEY_LOCAL="${OPENAI_API_KEY_LOCAL:-EMPTY}"
PROMPT100_FILE="${PROMPT100_FILE:-data/processed/rrd_sample_prompt_subset_100.ids.txt}"
REMAINING_PROMPT_ID_FILE="${REMAINING_PROMPT_ID_FILE:-data/processed/rrd_prompt_subset_except_prompt100.ids.txt}"

LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs}"
RUN_TAG="${RUN_TAG:-$(date '+%Y%m%d_%H%M%S')}"
LAUNCH_LOG="${LAUNCH_LOG:-${LOG_DIR}/gemma-large-rrd-remaining-${RUN_TAG}.log}"
SERVER_LOG="${SERVER_LOG:-${LOG_DIR}/gemma-large-rrd-remaining-${RUN_TAG}.server.log}"

mkdir -p "${LOG_DIR}"
: >>"${LAUNCH_LOG}"
exec >>"${LAUNCH_LOG}" 2>&1

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

build_remaining_prompt_file() {
    "${PYTHON_BIN}" - <<PY
import json
from pathlib import Path

prompt100_path = Path(${PROMPT100_FILE@Q})
if not prompt100_path.exists():
    raise SystemExit(f"Missing prompt100 subset file: {prompt100_path}")
prompt100_ids = {
    line.strip()
    for line in prompt100_path.read_text(encoding="utf-8").splitlines()
    if line.strip()
}
if len(prompt100_ids) != 100:
    raise SystemExit(f"Expected 100 prompt ids in {prompt100_path}, found {len(prompt100_ids)}")

pair_path = Path(${REPO_ROOT@Q}) / "data/processed/meta_eval_pairs.jsonl"
all_prompt_ids = []
seen = set()
with pair_path.open(encoding="utf-8") as handle:
    for line in handle:
        row = json.loads(line)
        prompt_id = row["prompt_id"]
        if prompt_id in seen:
            continue
        seen.add(prompt_id)
        all_prompt_ids.append(prompt_id)

remaining_ids = sorted(prompt_id for prompt_id in all_prompt_ids if prompt_id not in prompt100_ids)
output_path = Path(${REMAINING_PROMPT_ID_FILE@Q})
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text("".join(f"{prompt_id}\n" for prompt_id in remaining_ids), encoding="utf-8")
print(f"Wrote remaining prompt subset: {output_path} ({len(remaining_ids)} prompts)")
PY
}

backup_outputs() {
    local stamp
    stamp="$(date '+%Y%m%d_%H%M%S')"
    local -a files=(
        "data/processed/rubrics_rrd_pairwise_reference__${GENERATOR_FAMILY}.jsonl"
        "data/processed/rubrics_rrd_pairwise_reference__${GENERATOR_FAMILY}.jsonl.progress.json"
        "data/processed/rubrics_rrd_pairwise_sample__${GENERATOR_FAMILY}.jsonl"
        "data/processed/rubrics_rrd_pairwise_sample__${GENERATOR_FAMILY}.jsonl.progress.json"
    )
    for path in "${files[@]}"; do
        if [[ -f "${path}" ]]; then
            cp "${path}" "${path}.remaining_backup_${stamp}"
        fi
    done
}

gpu_target_pids() {
    if command -v lsof >/dev/null 2>&1; then
        lsof /dev/nvidia0 /dev/nvidia1 /dev/nvidia2 /dev/nvidia3 /dev/nvidiactl /dev/nvidia-uvm 2>/dev/null \
            | awk 'NR > 1 {print $2}' \
            | sort -u \
            | tr '\n' ' ' \
            | sed 's/[[:space:]]*$//'
        return 0
    fi
    pgrep -f 'vllm\.entrypoints\.openai\.api_server|VLLM::EngineCore|VLLM::Eng|VLLM::Wor' \
        | sort -u \
        | tr '\n' ' ' \
        | sed 's/[[:space:]]*$//'
}

stop_gpu_models() {
    local pids=""
    pids="$(gpu_target_pids)"
    if [[ -z "${pids}" ]]; then
        status "No running compute processes found on GPUs 0,1,2,3"
        return 0
    fi
    status "Stopping compute processes on GPUs 0,1,2,3: ${pids}"
    kill ${pids} 2>/dev/null || true
    sleep 8
    local remaining=()
    local pid
    for pid in ${pids}; do
        if kill -0 "${pid}" 2>/dev/null; then
            remaining+=("${pid}")
        fi
    done
    if (( ${#remaining[@]} > 0 )); then
        status "Force killing lingering GPU processes: ${remaining[*]}"
        kill -9 "${remaining[@]}" 2>/dev/null || true
        sleep 3
    fi
}

wait_for_api_ready() {
    local waited=0
    while (( waited < 1800 )); do
        if curl -fsS "http://127.0.0.1:${GEMMA_PORT}/health" >/dev/null 2>&1 \
            && curl -fsS "http://127.0.0.1:${GEMMA_PORT}/v1/models" | grep -F "\"${GEMMA_SERVED_NAME}\"" >/dev/null 2>&1; then
            status "vLLM ready on port=${GEMMA_PORT} model=${GEMMA_SERVED_NAME}"
            return 0
        fi
        if [[ -n "${SERVER_PID}" ]] && ! kill -0 "${SERVER_PID}" 2>/dev/null; then
            status "vLLM exited during startup"
            tail -n 120 "${SERVER_LOG}" || true
            return 1
        fi
        sleep 5
        waited=$((waited + 5))
    done
    status "Timed out waiting for vLLM readiness on port=${GEMMA_PORT}"
    tail -n 120 "${SERVER_LOG}" || true
    return 1
}

server_healthy() {
    curl -fsS "http://127.0.0.1:${GEMMA_PORT}/health" >/dev/null 2>&1
}

start_server() {
    local old_pids=""
    old_pids="$(lsof -tiTCP:${GEMMA_PORT} -sTCP:LISTEN 2>/dev/null || true)"
    if [[ -n "${old_pids}" ]]; then
        status "Stopping existing listeners on port=${GEMMA_PORT}: ${old_pids}"
        kill ${old_pids} 2>/dev/null || true
        sleep 2
    fi

    stop_gpu_models

    status "Starting ${GENERATOR_FAMILY} vLLM model=${GEMMA_MODEL_NAME} cuda=${GEMMA_CUDA_DEVICES} tp=${GEMMA_TENSOR_PARALLEL_SIZE} port=${GEMMA_PORT}"
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
        >"${SERVER_LOG}" 2>&1 &
    SERVER_PID=$!
    wait_for_api_ready
}

ensure_server() {
    if server_healthy; then
        return 0
    fi
    cleanup
    start_server
}

remaining_target_rows() {
    local method="$1"
    "${PYTHON_BIN}" - <<PY
import json
from pathlib import Path

method = ${method@Q}
prompt_ids = {
    line.strip()
    for line in Path(${REMAINING_PROMPT_ID_FILE@Q}).read_text(encoding="utf-8").splitlines()
    if line.strip()
}
output_path = Path(${REPO_ROOT@Q}) / "data/processed" / f"rubrics_{method}__${GENERATOR_FAMILY}.jsonl"
existing_keys = set()
if output_path.exists():
    with output_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            key = row.get("pair_id") if method == "rrd_pairwise_reference" else row.get("prompt_id")
            if key:
                existing_keys.add(str(key))

if method == "rrd_pairwise_reference":
    pair_path = Path(${REPO_ROOT@Q}) / "data/processed/meta_eval_pairs.jsonl"
    remaining = 0
    with pair_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["prompt_id"] not in prompt_ids:
                continue
            if row["pair_id"] not in existing_keys:
                remaining += 1
    print(remaining)
else:
    remaining = sum(1 for prompt_id in prompt_ids if prompt_id not in existing_keys)
    print(remaining)
PY
}

run_generation_once() {
    local method="$1"
    local generation_log="${LOG_DIR}/${GENERATOR_FAMILY}-${method}-remaining-${RUN_TAG}.generate.log"
    status "Running ${method} for remaining prompts with no fallback retries"
    OPENAI_BASE_URL="http://127.0.0.1:${GEMMA_PORT}/v1" \
    OPENAI_API_KEY="${OPENAI_API_KEY_LOCAL}" \
    KCC_RUBRIC_GENERATION_BACKEND="openai_compatible" \
    KCC_EMBED_DEVICE="cpu" \
    "${PYTHON_BIN}" -m src.generate_rubrics \
        --config "${CONFIG_PATH}" \
        --generator_family "${GENERATOR_FAMILY}" \
        --method "${method}" \
        --concurrency "${CONCURRENCY}" \
        --prompt-id-file "${REMAINING_PROMPT_ID_FILE}" \
        --no-fallback \
        --max-row-attempts "${MAX_ROW_ATTEMPTS}" \
        >>"${generation_log}" 2>&1
}

run_method_until_complete() {
    local method="$1"
    local attempt=1
    local remaining_before=""
    local remaining_after=""

    while (( attempt <= MAX_METHOD_ATTEMPTS )); do
        remaining_before="$(remaining_target_rows "${method}")"
        if [[ "${remaining_before}" == "0" ]]; then
            status "${method} already complete for remaining prompts"
            return 0
        fi

        status "Starting ${method} attempt=${attempt}/${MAX_METHOD_ATTEMPTS} remaining_target_rows=${remaining_before}"
        ensure_server
        if run_generation_once "${method}"; then
            remaining_after="$(remaining_target_rows "${method}")"
            status "${method} attempt=${attempt} finished remaining_target_rows=${remaining_after}"
            if [[ "${remaining_after}" == "0" ]]; then
                return 0
            fi
        else
            remaining_after="$(remaining_target_rows "${method}")"
            status "${method} attempt=${attempt} failed remaining_target_rows=${remaining_after}"
            tail -n 80 "${LOG_DIR}/${GENERATOR_FAMILY}-${method}-remaining-${RUN_TAG}.generate.log" || true
        fi

        if [[ "${remaining_after}" == "${remaining_before}" ]]; then
            status "No progress detected for ${method} on attempt=${attempt}; restarting vLLM before retry"
            cleanup
        fi
        sleep 5
        attempt=$((attempt + 1))
    done

    status "Exhausted method retries for ${method}"
    return 1
}

verify_final_outputs() {
    "${PYTHON_BIN}" - <<PY
import json
from pathlib import Path

root = Path(${REPO_ROOT@Q})
pair_rows = [json.loads(line) for line in (root / "data/processed/meta_eval_pairs.jsonl").open(encoding="utf-8") if line.strip()]
full_pair_count = len(pair_rows)
full_prompt_ids = {row["prompt_id"] for row in pair_rows}
full_prompt_count = len(full_prompt_ids)

checks = [
    ("rrd_pairwise_reference", full_pair_count, "pair_id"),
    ("rrd_pairwise_sample", full_prompt_count, "prompt_id"),
]
errors = []
for method, expected_rows, key_field in checks:
    path = root / "data/processed" / f"rubrics_{method}__${GENERATOR_FAMILY}.jsonl"
    if not path.exists():
        errors.append(f"missing:{path}")
        continue
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    if len(rows) != expected_rows:
        errors.append(f"count:{path.name}:{len(rows)}!={expected_rows}")
    keys = [str(row.get(key_field, "")) for row in rows]
    if len(keys) != len(set(keys)):
        errors.append(f"duplicate_keys:{path.name}")
    fallback_rows = [
        row.get(key_field, "<missing>")
        for row in rows
        if row.get("generation_metadata", {}).get("fallback_used")
        or row.get("generation_metadata", {}).get("recovered_from_partial_rrd")
        or row.get("generation_metadata", {}).get("weighting", {}).get("fallback_mode")
    ]
    if fallback_rows:
        errors.append(f"fallback:{path.name}:{len(fallback_rows)}")

if errors:
    raise SystemExit("\n".join(errors))
print(
    f"Verified gemma_large full outputs: "
    f"rrd_pairwise_reference={full_pair_count}, rrd_pairwise_sample={full_prompt_count}, fallback=0"
)
PY
}

status "Preparing remaining gemma_large RRD generation"
status "Python=${PYTHON_BIN}"
status "vLLM Python=${GEMMA_VLLM_PYTHON_BIN}"
status "Config=${CONFIG_PATH}"
status "Logs: launch=${LAUNCH_LOG} server=${SERVER_LOG}"
build_remaining_prompt_file
backup_outputs
start_server
run_method_until_complete "rrd_pairwise_reference"
run_method_until_complete "rrd_pairwise_sample"
verify_final_outputs
status "Completed remaining gemma_large RRD generation without saved fallback rows"
