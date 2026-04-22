#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
QWEN_VLLM_PYTHON_BIN="${QWEN_VLLM_PYTHON_BIN:-/tmp/venvs/rrd-qwen-vllm/bin/python}"
GEMMA_VLLM_PYTHON_BIN="${GEMMA_VLLM_PYTHON_BIN:-/tmp/venvs/vllm-gemma4-019/bin/python}"
EVAL_VLLM_PYTHON_BIN="${EVAL_VLLM_PYTHON_BIN:-/tmp/venvs/rrd-qwen-vllm/bin/python}"
CONCURRENCY="${CONCURRENCY:-30}"
GEN_GPU_MEMORY_UTILIZATION="${GEN_GPU_MEMORY_UTILIZATION:-0.90}"
EVAL_GPU_MEMORY_UTILIZATION="${EVAL_GPU_MEMORY_UTILIZATION:-0.92}"
GEN_MAX_MODEL_LEN="${GEN_MAX_MODEL_LEN:-32768}"
OPENAI_API_KEY_LOCAL="${OPENAI_API_KEY_LOCAL:-EMPTY}"
KEEP_SERVERS="${KEEP_SERVERS:-0}"
TARGET_FAMILIES="${TARGET_FAMILIES:-qwen_small qwen_large gemma_small gemma_large}"

LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs}"
RUN_TAG="${RUN_TAG:-$(date '+%Y%m%d_%H%M%S')}"
LAUNCH_LOG="${LAUNCH_LOG:-${LOG_DIR}/rubrics-vllm-launch-${RUN_TAG}.log}"
mkdir -p "${LOG_DIR}"
: >>"${LAUNCH_LOG}"
exec >>"${LAUNCH_LOG}" 2>&1

declare -A SERVER_PIDS
declare -A GEN_PIDS
PROGRESS_MONITOR_PID=""
EVAL_SERVER_PID=""
GENERATION_FAILED=0

status() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

cleanup() {
    if [[ -n "${PROGRESS_MONITOR_PID}" ]]; then
        kill "${PROGRESS_MONITOR_PID}" 2>/dev/null || true
        wait "${PROGRESS_MONITOR_PID}" 2>/dev/null || true
    fi
    if [[ "${KEEP_SERVERS}" != "1" ]]; then
        for pid in "${SERVER_PIDS[@]:-}"; do
            kill "${pid}" 2>/dev/null || true
            wait "${pid}" 2>/dev/null || true
        done
        if [[ -n "${EVAL_SERVER_PID}" ]]; then
            kill "${EVAL_SERVER_PID}" 2>/dev/null || true
            wait "${EVAL_SERVER_PID}" 2>/dev/null || true
        fi
    fi
}
trap cleanup EXIT INT TERM

start_server() {
    local family="$1"
    local model_name="$2"
    local gpu="$3"
    local port="$4"
    local python_bin="$5"
    local log_file="${LOG_DIR}/${family}-vllm-${RUN_TAG}.server.log"

    status "Starting ${family} vLLM model=${model_name} gpu=${gpu} port=${port}"
    CUDA_VISIBLE_DEVICES="${gpu}" \
    OPENAI_API_KEY="${OPENAI_API_KEY_LOCAL}" \
    "${python_bin}" -m vllm.entrypoints.openai.api_server \
        --model "${model_name}" \
        --served-model-name "${model_name}" \
        --port "${port}" \
        --gpu-memory-utilization "${GEN_GPU_MEMORY_UTILIZATION}" \
        --max-model-len "${GEN_MAX_MODEL_LEN}" \
        --tensor-parallel-size 1 \
        --trust-remote-code \
        >"${log_file}" 2>&1 &
    local pid=$!
    SERVER_PIDS["${family}"]="${pid}"

    local waited=0
    while (( waited < 1800 )); do
        if curl -fsS "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
            status "${family} vLLM ready"
            return 0
        fi
        if ! kill -0 "${pid}" 2>/dev/null; then
            status "${family} vLLM failed during startup"
            tail -n 80 "${log_file}" || true
            return 1
        fi
        sleep 5
        waited=$((waited + 5))
    done
    status "${family} vLLM timed out during startup"
    tail -n 80 "${log_file}" || true
    return 1
}

start_generation() {
    local family="$1"
    local port="$2"
    local log_file="${LOG_DIR}/${family}-generate-${RUN_TAG}.log"
    status "Starting rubric generation family=${family} concurrency=${CONCURRENCY}"
    OPENAI_BASE_URL="http://127.0.0.1:${port}/v1" \
    OPENAI_API_KEY="${OPENAI_API_KEY_LOCAL}" \
    KCC_EMBED_DEVICE="cpu" \
    "${PYTHON_BIN}" -m src.generate_rubrics --config configs/main.yaml --generator_family "${family}" --concurrency "${CONCURRENCY}" \
        >"${log_file}" 2>&1 &
    GEN_PIDS["${family}"]=$!
}

start_progress_monitor() {
    local log_file="${LOG_DIR}/rubric-progress-${RUN_TAG}.log"
    status "Starting progress monitor log=${log_file}"
    (
        while true; do
            printf '\n[%s]\n' "$(date '+%Y-%m-%d %H:%M:%S')"
            bash "${REPO_ROOT}/scripts/show_rubric_progress.sh" || true
            sleep 60
        done
    ) >"${log_file}" 2>&1 &
    PROGRESS_MONITOR_PID=$!
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

wait_for_generation() {
    for family in "${!GEN_PIDS[@]}"; do
        local pid="${GEN_PIDS[${family}]}"
        if wait "${pid}"; then
            status "Generation finished family=${family}"
        else
            if family_output_complete "${family}" >/dev/null 2>&1; then
                status "Generation for family=${family} exited non-zero, but all expected outputs are complete; treating as success"
            else
                status "Generation failed family=${family}"
                GENERATION_FAILED=1
            fi
        fi
    done
    return 0
}

stop_generation_servers() {
    for family in "${!SERVER_PIDS[@]}"; do
        local pid="${SERVER_PIDS[${family}]}"
        status "Stopping ${family} vLLM pid=${pid}"
        kill "${pid}" 2>/dev/null || true
        wait "${pid}" 2>/dev/null || true
        unset "SERVER_PIDS[${family}]"
    done
}

verify_generation_outputs() {
    status "Verifying rubric outputs"
    "${PYTHON_BIN}" - <<'PY'
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

errors = []
for spec in get_condition_specs(config):
    if spec.rubric_scope == "none":
        continue
    path = root / "data/processed" / f"rubrics_{spec.method}__{spec.generator_family}.jsonl"
    if not path.exists():
        errors.append(f"missing:{path}")
        continue
    row_count = sum(1 for line in path.open(encoding="utf-8") if line.strip())
    expected = pair_count if spec.rubric_scope == "pair" else prompt_count
    if row_count != expected:
        errors.append(f"count:{path}:{row_count}!={expected}")
if errors:
    print("\n".join(errors))
    raise SystemExit(1)
print("OK")
PY
}

start_eval_server() {
    local port="${1:-8012}"
    local log_file="${LOG_DIR}/gpt-oss-eval-${RUN_TAG}.server.log"
    status "Starting gpt-oss-120b vLLM on GPUs 0,1,2,3 port=${port}"
    CUDA_VISIBLE_DEVICES="0,1,2,3" \
    OPENAI_API_KEY="${OPENAI_API_KEY_LOCAL}" \
    "${EVAL_VLLM_PYTHON_BIN}" -m vllm.entrypoints.openai.api_server \
        --model openai/gpt-oss-120b \
        --served-model-name gpt-oss-120b \
        --port "${port}" \
        --gpu-memory-utilization "${EVAL_GPU_MEMORY_UTILIZATION}" \
        --tensor-parallel-size 4 \
        --trust-remote-code \
        >"${log_file}" 2>&1 &
    EVAL_SERVER_PID=$!

    local waited=0
    while (( waited < 1800 )); do
        if curl -fsS "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
            status "gpt-oss-120b vLLM ready"
            return 0
        fi
        if ! kill -0 "${EVAL_SERVER_PID}" 2>/dev/null; then
            status "gpt-oss-120b vLLM failed during startup"
            tail -n 120 "${log_file}" || true
            return 1
        fi
        sleep 5
        waited=$((waited + 5))
    done
    status "gpt-oss-120b vLLM timed out during startup"
    tail -n 120 "${log_file}" || true
    return 1
}

run_eval() {
    local log_file="${LOG_DIR}/pairwise-eval-vllm-${RUN_TAG}.log"
    status "Running pairwise evaluation"
    OPENAI_BASE_URL="http://127.0.0.1:8012/v1" \
    OPENAI_API_KEY="${OPENAI_API_KEY_LOCAL}" \
    "${PYTHON_BIN}" -m src.evaluate_pairs --config configs/main.yaml \
        >"${log_file}" 2>&1
    OPENAI_BASE_URL="http://127.0.0.1:8012/v1" \
    OPENAI_API_KEY="${OPENAI_API_KEY_LOCAL}" \
    "${PYTHON_BIN}" -m src.export_run_manifest --config configs/main.yaml \
        >>"${log_file}" 2>&1
    status "Evaluation finished"
}

status "Launcher starting"
status "Logs directory: ${LOG_DIR}"

for family in ${TARGET_FAMILIES}; do
    case "${family}" in
        qwen_small)
            start_server qwen_small "Qwen/Qwen3-4B-FP8" 0 8101 "${QWEN_VLLM_PYTHON_BIN}"
            ;;
        qwen_large)
            start_server qwen_large "Qwen/Qwen3-32B" 1 8102 "${QWEN_VLLM_PYTHON_BIN}"
            ;;
        gemma_small)
            start_server gemma_small "google/gemma-4-E4B-it" 2 8103 "${GEMMA_VLLM_PYTHON_BIN}"
            ;;
        gemma_large)
            start_server gemma_large "google/gemma-4-31b-it" 3 8104 "${GEMMA_VLLM_PYTHON_BIN}"
            ;;
        *)
            echo "Unknown family: ${family}" >&2
            exit 1
            ;;
    esac
done

for family in ${TARGET_FAMILIES}; do
    case "${family}" in
        qwen_small) start_generation qwen_small 8101 ;;
        qwen_large) start_generation qwen_large 8102 ;;
        gemma_small) start_generation gemma_small 8103 ;;
        gemma_large) start_generation gemma_large 8104 ;;
    esac
done
start_progress_monitor

wait_for_generation
if [[ "${GENERATION_FAILED}" == "1" ]]; then
    status "At least one family failed; preserving completed outputs and skipping auto-eval"
    exit 1
fi
verify_generation_outputs
stop_generation_servers
start_eval_server 8012
run_eval

status "Launcher completed successfully"
