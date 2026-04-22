#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
VLLM_PYTHON_BIN="${VLLM_PYTHON_BIN:-/tmp/venvs/rrd-qwen-vllm/bin/python}"
MODEL_NAME="${MODEL_NAME:-openai/gpt-oss-120b}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-gpt-oss-120b}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-4}"
PORT="${PORT:-8012}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.92}"
SERVER_STARTUP_TIMEOUT_SECONDS="${SERVER_STARTUP_TIMEOUT_SECONDS:-1800}"
SERVER_POLL_INTERVAL_SECONDS="${SERVER_POLL_INTERVAL_SECONDS:-5}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-1}"
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"
KEEP_SERVER="${KEEP_SERVER:-0}"
OPENAI_API_KEY_LOCAL="${OPENAI_API_KEY_LOCAL:-EMPTY}"

LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs}"
RUN_TAG="${RUN_TAG:-$(date '+%Y%m%d_%H%M%S')}"
WATCH_LOG="${WATCH_LOG:-${LOG_DIR}/pairwise-watch-${RUN_TAG}.log}"
SERVER_LOG="${SERVER_LOG:-${LOG_DIR}/pairwise-vllm-${RUN_TAG}.server.log}"
EVAL_LOG="${EVAL_LOG:-${LOG_DIR}/pairwise-eval-${RUN_TAG}.log}"

mkdir -p "${LOG_DIR}"
: >>"${WATCH_LOG}"
: >>"${SERVER_LOG}"
: >>"${EVAL_LOG}"
exec >>"${WATCH_LOG}" 2>&1

CURRENT_SERVER_PID=""

status() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

cleanup() {
    if [[ -n "${CURRENT_SERVER_PID}" && "${KEEP_SERVER}" != "1" ]]; then
        status "Stopping vLLM server pid=${CURRENT_SERVER_PID}"
        kill "${CURRENT_SERVER_PID}" 2>/dev/null || true
        wait "${CURRENT_SERVER_PID}" 2>/dev/null || true
        CURRENT_SERVER_PID=""
    fi
}
trap cleanup EXIT INT TERM

rubric_process_count() {
    pgrep -af "${REPO_ROOT}/.venv/bin/python -m src.generate_rubrics" | wc -l | tr -d ' '
}

wait_for_rubrics_to_finish() {
    status "Waiting for running rubric generation processes to finish"
    while true; do
        local count
        count="$(rubric_process_count)"
        if [[ "${count}" == "0" ]]; then
            status "No active rubric generation process remains"
            return 0
        fi
        status "Rubric generation still running: count=${count}"
        sleep 30
    done
}

verify_rubric_outputs() {
    status "Verifying expected rubric output files and row counts"
    "${PYTHON_BIN}" - <<'PY'
import json
from pathlib import Path
import sys

root = Path("/data/minseo/KCC2026/korean-rubric-grounding-main")
sys.path.insert(0, str(root))

from src.utils import get_condition_specs, get_meta_eval_pairs_path, load_config  # noqa: E402

config = load_config(root / "configs/main.yaml")
pairs_path = get_meta_eval_pairs_path()
pairs = [json.loads(line) for line in pairs_path.open(encoding="utf-8") if line.strip()]
pair_count = len(pairs)
prompt_count = len({row["prompt_id"] for row in pairs})

missing: list[str] = []
bad: list[str] = []
for spec in get_condition_specs(config):
    if spec.rubric_scope == "none":
        continue
    path = root / "data/processed" / f"rubrics_{spec.method}__{spec.generator_family}.jsonl"
    if not path.exists():
        missing.append(str(path))
        continue
    row_count = sum(1 for line in path.open(encoding="utf-8") if line.strip())
    expected = pair_count if spec.rubric_scope == "pair" else prompt_count
    if row_count != expected:
        bad.append(f"{path}:{row_count}!={expected}")

if missing or bad:
    if missing:
        print("MISSING")
        for item in missing:
            print(item)
    if bad:
        print("BAD")
        for item in bad:
            print(item)
    raise SystemExit(1)

print(f"OK pair_count={pair_count} prompt_count={prompt_count}")
PY
}

start_vllm() {
    export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
    export OPENAI_BASE_URL="http://127.0.0.1:${PORT}/v1"
    export OPENAI_API_BASE="${OPENAI_BASE_URL}"
    export OPENAI_API_KEY="${OPENAI_API_KEY_LOCAL}"
    export VLLM_BASE_URL="${OPENAI_BASE_URL}"
    export VLLM_API_KEY="${OPENAI_API_KEY_LOCAL}"

    status "Starting vLLM model=${MODEL_NAME} served_model=${SERVED_MODEL_NAME} gpus=${CUDA_VISIBLE_DEVICES} port=${PORT}"
    local -a server_cmd=(
        "${VLLM_PYTHON_BIN}"
        -m
        vllm.entrypoints.openai.api_server
        --model "${MODEL_NAME}"
        --served-model-name "${SERVED_MODEL_NAME}"
        --port "${PORT}"
        --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
        --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
    )
    if [[ "${TRUST_REMOTE_CODE}" == "1" ]]; then
        server_cmd+=(--trust-remote-code)
    fi
    if [[ -n "${VLLM_EXTRA_ARGS}" ]]; then
        # shellcheck disable=SC2206
        local extra_args=( ${VLLM_EXTRA_ARGS} )
        server_cmd+=("${extra_args[@]}")
    fi

    {
        set -x
        "${server_cmd[@]}"
    } >"${SERVER_LOG}" 2>&1 &

    CURRENT_SERVER_PID=$!

    local waited=0
    while (( waited < SERVER_STARTUP_TIMEOUT_SECONDS )); do
        if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
            status "vLLM ready base_url=${OPENAI_BASE_URL}"
            return 0
        fi
        if ! kill -0 "${CURRENT_SERVER_PID}" 2>/dev/null; then
            status "vLLM exited during startup; tailing server log"
            tail -n 120 "${SERVER_LOG}" || true
            return 1
        fi
        sleep "${SERVER_POLL_INTERVAL_SECONDS}"
        waited=$((waited + SERVER_POLL_INTERVAL_SECONDS))
    done

    status "Timed out waiting for vLLM health endpoint"
    tail -n 120 "${SERVER_LOG}" || true
    return 1
}

run_pairwise_eval() {
    status "Starting pairwise evaluation via local vLLM"
    {
        set -x
        OPENAI_BASE_URL="http://127.0.0.1:${PORT}/v1" \
        OPENAI_API_KEY="${OPENAI_API_KEY_LOCAL}" \
        "${PYTHON_BIN}" -m src.evaluate_pairs --config configs/main.yaml
        OPENAI_BASE_URL="http://127.0.0.1:${PORT}/v1" \
        OPENAI_API_KEY="${OPENAI_API_KEY_LOCAL}" \
        "${PYTHON_BIN}" -m src.export_run_manifest --config configs/main.yaml
    } >>"${EVAL_LOG}" 2>&1
    status "Pairwise evaluation and manifest export completed"
}

status "Watcher starting"
status "Logs: watch=${WATCH_LOG} server=${SERVER_LOG} eval=${EVAL_LOG}"
wait_for_rubrics_to_finish
verify_rubric_outputs
start_vllm
run_pairwise_eval
status "Watcher finished successfully"
