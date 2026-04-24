#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

DEFAULT_PYTHON_BIN="$(command -v python3 || command -v python)"
PYTHON_BIN="${PYTHON_BIN:-${DEFAULT_PYTHON_BIN}}"

QWEN_VLLM_PYTHON_BIN="${QWEN_VLLM_PYTHON_BIN:-${DEFAULT_PYTHON_BIN}}"
GEMMA_VLLM_PYTHON_BIN="${GEMMA_VLLM_PYTHON_BIN:-/data/minseo/envs/vllm-gemma4-019/bin/python}"

QWEN_MODEL_NAME="${QWEN_MODEL_NAME:-Qwen/Qwen3-4B-FP8}"
QWEN_SERVED_NAME="${QWEN_SERVED_NAME:-Qwen/Qwen3-4B-FP8}"
QWEN_PORT="${QWEN_PORT:-8110}"
QWEN_CUDA_DEVICES="${QWEN_CUDA_DEVICES:-0,1}"
QWEN_TP_SIZE="${QWEN_TP_SIZE:-2}"
QWEN_GPU_MEMORY_UTILIZATION="${QWEN_GPU_MEMORY_UTILIZATION:-0.80}"
QWEN_MAX_MODEL_LEN="${QWEN_MAX_MODEL_LEN:-32768}"
QWEN_MAX_NUM_SEQS="${QWEN_MAX_NUM_SEQS:-4}"
QWEN_CONCURRENCY="${QWEN_CONCURRENCY:-4}"
QWEN_CONFIG_PATH="${QWEN_CONFIG_PATH:-configs/rrd_sample_prompt100_small.yaml}"

GEMMA_MODEL_NAME="${GEMMA_MODEL_NAME:-google/gemma-4-E4B-it}"
GEMMA_SERVED_NAME="${GEMMA_SERVED_NAME:-google/gemma-4-E4B-it}"
GEMMA_PORT="${GEMMA_PORT:-8111}"
GEMMA_CUDA_DEVICES="${GEMMA_CUDA_DEVICES:-2,3}"
GEMMA_TP_SIZE="${GEMMA_TP_SIZE:-2}"
GEMMA_GPU_MEMORY_UTILIZATION="${GEMMA_GPU_MEMORY_UTILIZATION:-0.80}"
GEMMA_MAX_MODEL_LEN="${GEMMA_MAX_MODEL_LEN:-32768}"
GEMMA_MAX_NUM_SEQS="${GEMMA_MAX_NUM_SEQS:-4}"
GEMMA_CONCURRENCY="${GEMMA_CONCURRENCY:-4}"
GEMMA_CONFIG_PATH="${GEMMA_CONFIG_PATH:-configs/rrd_sample_prompt100_gemma_e4b.yaml}"

OPENAI_API_KEY_LOCAL="${OPENAI_API_KEY_LOCAL:-EMPTY}"
PROMPT100_ID_FILE="${PROMPT100_ID_FILE:-data/processed/rrd_sample_prompt_subset_100.ids.txt}"
REMAINING_ID_FILE="${REMAINING_ID_FILE:-data/processed/rrd_remaining_prompt_subset_473.ids.txt}"
PROMPT100_SPLIT="${PROMPT100_SPLIT:-prompt100}"
REMAINING_SPLIT="${REMAINING_SPLIT:-remaining473}"
MAX_REPAIR_ATTEMPTS="${MAX_REPAIR_ATTEMPTS:-5}"

LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs}"
RUN_TAG="${RUN_TAG:-$(date '+%Y%m%d_%H%M%S')}"
MASTER_LOG="${MASTER_LOG:-${LOG_DIR}/rrd-remaining-qwen-gemma-${RUN_TAG}.log}"

mkdir -p "${LOG_DIR}"
: >>"${MASTER_LOG}"
exec >>"${MASTER_LOG}" 2>&1

QWEN_PID=""
GEMMA_PID=""

status() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >&2
}

cleanup() {
    if [[ -n "${QWEN_PID}" ]]; then
        kill "${QWEN_PID}" 2>/dev/null || true
        wait "${QWEN_PID}" 2>/dev/null || true
        QWEN_PID=""
    fi
    if [[ -n "${GEMMA_PID}" ]]; then
        kill "${GEMMA_PID}" 2>/dev/null || true
        wait "${GEMMA_PID}" 2>/dev/null || true
        GEMMA_PID=""
    fi
}
trap cleanup EXIT INT TERM

ensure_remaining_subset() {
    "${PYTHON_BIN}" - <<PY
from pathlib import Path
import json

repo_root = Path(${REPO_ROOT@Q})
base = repo_root / "data/processed"
subset_path = Path(${PROMPT100_ID_FILE@Q})
if not subset_path.is_absolute():
    subset_path = repo_root / subset_path
remaining_path = Path(${REMAINING_ID_FILE@Q})
if not remaining_path.is_absolute():
    remaining_path = repo_root / remaining_path
pairs_path = base / "meta_eval_pairs.jsonl"

subset = {
    line.strip()
    for line in subset_path.read_text(encoding="utf-8").splitlines()
    if line.strip()
}
all_prompts = []
seen = set()
with pairs_path.open(encoding="utf-8") as handle:
    for line in handle:
        row = json.loads(line)
        prompt_id = row["prompt_id"]
        if prompt_id not in seen:
            seen.add(prompt_id)
            all_prompts.append(prompt_id)
remaining = sorted(prompt_id for prompt_id in all_prompts if prompt_id not in subset)
remaining_path.write_text("".join(f"{prompt_id}\n" for prompt_id in remaining), encoding="utf-8")
print(f"remaining_prompt_count={len(remaining)}")
PY
}

wait_for_openai_ready() {
    local family="$1"
    local port="$2"
    local pid="$3"
    local log_file="$4"
    local served_name="$5"
    local waited=0

    while (( waited < 3600 )); do
        if curl -fsS "http://127.0.0.1:${port}/v1/models" | grep -F "\"${served_name}\"" >/dev/null 2>&1; then
            local payload=""
            if [[ "${family}" == "qwen_small" ]]; then
                payload='{"model":"'"${served_name}"'","messages":[{"role":"user","content":"ping"}],"max_tokens":8,"temperature":0,"extra_body":{"chat_template_kwargs":{"enable_thinking":false}}}'
            else
                payload='{"model":"'"${served_name}"'","messages":[{"role":"user","content":"ping"}],"max_tokens":8,"temperature":0}'
            fi
            local completion_code=""
            completion_code="$(
                curl -sS -o "/tmp/${family}_preflight_${port}.json" -w '%{http_code}' \
                    "http://127.0.0.1:${port}/v1/chat/completions" \
                    -H 'Content-Type: application/json' \
                    -H "Authorization: Bearer ${OPENAI_API_KEY_LOCAL}" \
                    -d "${payload}" || true
            )"
            if [[ "${completion_code}" == "200" ]]; then
                status "[${family}] vLLM ready on port=${port}"
                return 0
            fi
        fi
        if ! kill -0 "${pid}" 2>/dev/null; then
            status "[${family}] vLLM exited during startup"
            tail -n 120 "${log_file}" || true
            return 1
        fi
        sleep 5
        waited=$((waited + 5))
    done

    status "[${family}] timed out waiting for readiness on port=${port}"
    tail -n 120 "${log_file}" || true
    return 1
}

start_server() {
    local family="$1"
    local python_bin="$2"
    local model_name="$3"
    local served_name="$4"
    local port="$5"
    local cuda_devices="$6"
    local tp_size="$7"
    local gpu_memory_utilization="$8"
    local max_model_len="$9"
    local max_num_seqs="${10}"
    local server_log="${LOG_DIR}/${family}-${REMAINING_SPLIT}-${RUN_TAG}.server.log"

    local listeners=""
    listeners="$(lsof -tiTCP:${port} -sTCP:LISTEN 2>/dev/null || true)"
    if [[ -n "${listeners}" ]]; then
        status "[${family}] killing stale listeners on port=${port}: ${listeners}"
        kill ${listeners} 2>/dev/null || true
        sleep 2
    fi

    status "[${family}] starting server model=${model_name} cuda=${cuda_devices} tp=${tp_size} port=${port}"
    CUDA_VISIBLE_DEVICES="${cuda_devices}" \
    OPENAI_API_KEY="${OPENAI_API_KEY_LOCAL}" \
    "${python_bin}" -m vllm.entrypoints.openai.api_server \
        --model "${model_name}" \
        --served-model-name "${served_name}" \
        --port "${port}" \
        --tensor-parallel-size "${tp_size}" \
        --gpu-memory-utilization "${gpu_memory_utilization}" \
        --max-model-len "${max_model_len}" \
        --max-num-seqs "${max_num_seqs}" \
        --enforce-eager \
        --trust-remote-code \
        >"${server_log}" 2>&1 &

    local pid="$!"
    wait_for_openai_ready "${family}" "${port}" "${pid}" "${server_log}" "${served_name}"
    printf '%s\n' "${pid}"
}

stop_server_pid() {
    local family="$1"
    local pid="$2"
    if [[ -n "${pid}" ]]; then
        status "[${family}] stopping server pid=${pid}"
        kill "${pid}" 2>/dev/null || true
        wait "${pid}" 2>/dev/null || true
    fi
}

run_generation() {
    local family="$1"
    local config_path="$2"
    local method="$3"
    local port="$4"
    local concurrency="$5"
    local prompt_id_file="$6"
    local output_split="$7"
    local generation_log="${LOG_DIR}/${family}-${method}-${output_split}-${RUN_TAG}.generate.log"

    status "[${family}/${method}] generating split=${output_split} prompt_ids=${prompt_id_file}"
    OPENAI_BASE_URL="http://127.0.0.1:${port}/v1" \
    OPENAI_API_KEY="${OPENAI_API_KEY_LOCAL}" \
    KCC_RUBRIC_GENERATION_BACKEND="openai_compatible" \
    KCC_EMBED_DEVICE="cpu" \
    "${PYTHON_BIN}" -u -m src.generate_rubrics \
        --config "${config_path}" \
        --generator_family "${family}" \
        --method "${method}" \
        --concurrency "${concurrency}" \
        --prompt-id-file "${prompt_id_file}" \
        --output-split "${output_split}" \
        >"${generation_log}" 2>&1
}

effective_fallback_prompt_count() {
    local family="$1"
    local method="$2"
    local split_prefix="$3"
    local ids_out="$4"

    "${PYTHON_BIN}" - <<PY
from pathlib import Path
import json

base = Path(${REPO_ROOT@Q}) / "data/processed"
family = ${family@Q}
method = ${method@Q}
split_prefix = ${split_prefix@Q}
ids_out = Path(${ids_out@Q})
key_field = "pair_id" if method in {"rar_pairwise_reference", "rrd_pairwise_reference"} else "prompt_id"
rows_by_key = {}
for path in sorted(base.glob(f"rubrics_{method}__{family}__{split_prefix}*.jsonl")):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rows_by_key[str(row[key_field])] = row

fallback_prompt_ids = sorted(
    {
        row["prompt_id"]
        for row in rows_by_key.values()
        if row.get("generation_metadata", {}).get("fallback_used")
    }
)
if fallback_prompt_ids:
    ids_out.write_text("".join(f"{prompt_id}\n" for prompt_id in fallback_prompt_ids), encoding="utf-8")
else:
    ids_out.write_text("", encoding="utf-8")
print(len(fallback_prompt_ids))
PY
}

repair_method() {
    local family="$1"
    local config_path="$2"
    local method="$3"
    local port="$4"
    local concurrency="$5"
    local split_prefix="$6"

    local attempt=1
    while (( attempt <= MAX_REPAIR_ATTEMPTS )); do
        local ids_file="data/processed/${family}_${method}_${split_prefix}_fallback_attempt${attempt}.ids.txt"
        local fallback_count=""
        fallback_count="$(effective_fallback_prompt_count "${family}" "${method}" "${split_prefix}" "${ids_file}")"
        status "[${family}/${method}] effective fallback prompt count after attempt $((attempt - 1)): ${fallback_count}"
        if [[ "${fallback_count}" == "0" ]]; then
            return 0
        fi
        run_generation "${family}" "${config_path}" "${method}" "${port}" "${concurrency}" "${ids_file}" "${split_prefix}_refill${attempt}"
        attempt=$((attempt + 1))
    done

    local final_ids="data/processed/${family}_${method}_${split_prefix}_fallback_final.ids.txt"
    local final_fallback_count=""
    final_fallback_count="$(effective_fallback_prompt_count "${family}" "${method}" "${split_prefix}" "${final_ids}")"
    status "[${family}/${method}] fallback prompt count after max retries: ${final_fallback_count}"
}

run_family() {
    local family="$1"
    local vllm_python="$2"
    local config_path="$3"
    local model_name="$4"
    local served_name="$5"
    local port="$6"
    local cuda_devices="$7"
    local tp_size="$8"
    local gpu_memory_utilization="$9"
    local max_model_len="${10}"
    local max_num_seqs="${11}"
    local concurrency="${12}"

    local server_pid=""
    trap 'stop_server_pid "'"${family}"'" "$server_pid"' RETURN
    server_pid="$(start_server "${family}" "${vllm_python}" "${model_name}" "${served_name}" "${port}" "${cuda_devices}" "${tp_size}" "${gpu_memory_utilization}" "${max_model_len}" "${max_num_seqs}")"

    run_generation "${family}" "${config_path}" "rrd_pairwise_reference" "${port}" "${concurrency}" "${REMAINING_ID_FILE}" "${REMAINING_SPLIT}"
    repair_method "${family}" "${config_path}" "rrd_pairwise_reference" "${port}" "${concurrency}" "${REMAINING_SPLIT}"

    run_generation "${family}" "${config_path}" "rrd_pairwise_sample" "${port}" "${concurrency}" "${REMAINING_ID_FILE}" "${REMAINING_SPLIT}"
    repair_method "${family}" "${config_path}" "rrd_pairwise_sample" "${port}" "${concurrency}" "${REMAINING_SPLIT}"

    stop_server_pid "${family}" "${server_pid}"
    trap - RETURN
}

merge_final_output() {
    local family="$1"
    local method="$2"

    "${PYTHON_BIN}" - <<PY
from pathlib import Path
import json
import shutil
import time

base = Path(${REPO_ROOT@Q}) / "data/processed"
family = ${family@Q}
method = ${method@Q}
prompt100_split = ${PROMPT100_SPLIT@Q}
remaining_split = ${REMAINING_SPLIT@Q}
run_tag = ${RUN_TAG@Q}

key_field = "pair_id" if method in {"rar_pairwise_reference", "rrd_pairwise_reference"} else "prompt_id"
expected_total = 1396 if method in {"rar_pairwise_reference", "rrd_pairwise_reference"} else 573

prompt100_path = base / f"rubrics_{method}__{family}__{prompt100_split}.jsonl"
remaining_paths = sorted(base.glob(f"rubrics_{method}__{family}__{remaining_split}*.jsonl"))
full_path = base / f"rubrics_{method}__{family}.jsonl"
progress_path = full_path.with_suffix(full_path.suffix + ".progress.json")

if full_path.exists():
    backup_path = full_path.with_name(full_path.name + f".before_remaining_merge_{run_tag}.bak")
    shutil.copy2(full_path, backup_path)
    print(f"backup={backup_path}")
if progress_path.exists():
    backup_path = progress_path.with_name(progress_path.name + f".before_remaining_merge_{run_tag}.bak")
    shutil.copy2(progress_path, backup_path)
    print(f"backup={backup_path}")

rows_by_key = {}
for path in [prompt100_path, *remaining_paths]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rows_by_key[str(row[key_field])] = row

merged = [rows_by_key[key] for key in sorted(rows_by_key)]
fallback_count = sum(1 for row in merged if row.get("generation_metadata", {}).get("fallback_used"))
if len(merged) != expected_total:
    raise SystemExit(f"Expected {expected_total} rows for {method}/{family}, found {len(merged)}")

with full_path.open("w", encoding="utf-8") as handle:
    for row in merged:
        handle.write(json.dumps(row, ensure_ascii=False) + "\\n")

progress_path.write_text(
    json.dumps(
        {
            "generator_family": family,
            "method": method,
            "output_path": str(full_path),
            "key_field": key_field,
            "completed_rows": len(merged),
            "expected_rows": expected_total,
            "remaining_rows": 0,
            "complete": True,
            "finalized_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        ensure_ascii=False,
        indent=2,
    )
    + "\\n",
    encoding="utf-8",
)
print(f"merged={full_path} rows={len(merged)} fallback={fallback_count}")
PY
}

status "Preparing remaining-subset RRD generation and merge"
ensure_remaining_subset

status "Launching qwen_small on GPUs ${QWEN_CUDA_DEVICES} and gemma_small on GPUs ${GEMMA_CUDA_DEVICES}"
run_family \
    "qwen_small" \
    "${QWEN_VLLM_PYTHON_BIN}" \
    "${QWEN_CONFIG_PATH}" \
    "${QWEN_MODEL_NAME}" \
    "${QWEN_SERVED_NAME}" \
    "${QWEN_PORT}" \
    "${QWEN_CUDA_DEVICES}" \
    "${QWEN_TP_SIZE}" \
    "${QWEN_GPU_MEMORY_UTILIZATION}" \
    "${QWEN_MAX_MODEL_LEN}" \
    "${QWEN_MAX_NUM_SEQS}" \
    "${QWEN_CONCURRENCY}" &
QWEN_PID="$!"

run_family \
    "gemma_small" \
    "${GEMMA_VLLM_PYTHON_BIN}" \
    "${GEMMA_CONFIG_PATH}" \
    "${GEMMA_MODEL_NAME}" \
    "${GEMMA_SERVED_NAME}" \
    "${GEMMA_PORT}" \
    "${GEMMA_CUDA_DEVICES}" \
    "${GEMMA_TP_SIZE}" \
    "${GEMMA_GPU_MEMORY_UTILIZATION}" \
    "${GEMMA_MAX_MODEL_LEN}" \
    "${GEMMA_MAX_NUM_SEQS}" \
    "${GEMMA_CONCURRENCY}" &
GEMMA_PID="$!"

qwen_status=0
gemma_status=0
wait "${QWEN_PID}" || qwen_status=$?
QWEN_PID=""
wait "${GEMMA_PID}" || gemma_status=$?
GEMMA_PID=""

if (( qwen_status != 0 || gemma_status != 0 )); then
    status "Family run failed: qwen_status=${qwen_status} gemma_status=${gemma_status}"
    exit 1
fi

status "Merging prompt100 and remaining outputs into final full-set files"
merge_final_output "qwen_small" "rrd_pairwise_sample"
merge_final_output "qwen_small" "rrd_pairwise_reference"
merge_final_output "gemma_small" "rrd_pairwise_sample"
merge_final_output "gemma_small" "rrd_pairwise_reference"

status "Completed remaining-subset generation and final full-set merge"
