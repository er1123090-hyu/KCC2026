#!/usr/bin/env bash
set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}" || exit 1

LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/finalize-remaining-after-gemma-$(date '+%Y%m%d_%H%M%S').log}"
mkdir -p "${LOG_DIR}"
exec >>"${LOG_FILE}" 2>&1

log() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

gemma_completed_rows() {
    python3 - <<'PY'
import json
from pathlib import Path

path = Path("data/processed/rubrics_rrd_pairwise_sample__gemma_small__remaining473.jsonl.progress.json")
try:
    print(json.loads(path.read_text(encoding="utf-8")).get("completed_rows", 0))
except Exception:
    print(0)
PY
}

fallback_prompt_count() {
    local ids_out="$1"
    python3 - "${ids_out}" <<'PY'
import json
import sys
from pathlib import Path

ids_out = Path(sys.argv[1])
rows_by_prompt = {}
for path in sorted(Path("data/processed").glob("rubrics_rrd_pairwise_sample__gemma_small__remaining473*.jsonl")):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rows_by_prompt[str(row["prompt_id"])] = row

fallback_prompt_ids = sorted(
    prompt_id
    for prompt_id, row in rows_by_prompt.items()
    if row.get("generation_metadata", {}).get("fallback_used")
)
ids_out.write_text("".join(f"{prompt_id}\n" for prompt_id in fallback_prompt_ids), encoding="utf-8")
print(len(fallback_prompt_ids))
PY
}

merge_full_outputs() {
    python3 - <<'PY'
import json
import shutil
import time
from pathlib import Path

base = Path("data/processed")
run_tag = time.strftime("%Y%m%d_%H%M%S_finalizer")

for family in ["qwen_small", "gemma_small"]:
    for method in ["rrd_pairwise_sample", "rrd_pairwise_reference"]:
        key_field = "pair_id" if method.endswith("reference") else "prompt_id"
        expected_total = 1396 if method.endswith("reference") else 573
        prompt100_path = base / f"rubrics_{method}__{family}__prompt100.jsonl"
        remaining_paths = sorted(base.glob(f"rubrics_{method}__{family}__remaining473*.jsonl"))
        full_path = base / f"rubrics_{method}__{family}.jsonl"
        progress_path = full_path.with_suffix(full_path.suffix + ".progress.json")

        if full_path.exists():
            shutil.copy2(full_path, full_path.with_name(full_path.name + f".before_remaining_merge_{run_tag}.bak"))
        if progress_path.exists():
            shutil.copy2(progress_path, progress_path.with_name(progress_path.name + f".before_remaining_merge_{run_tag}.bak"))

        rows_by_key = {}
        for path in [prompt100_path, *remaining_paths]:
            if not path.exists():
                raise SystemExit(f"missing input: {path}")
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    rows_by_key[str(row[key_field])] = row

        merged = [rows_by_key[key] for key in sorted(rows_by_key)]
        fallback_count = sum(1 for row in merged if row.get("generation_metadata", {}).get("fallback_used"))
        if len(merged) != expected_total:
            raise SystemExit(f"{family}/{method}: expected {expected_total}, got {len(merged)}")
        if fallback_count:
            raise SystemExit(f"{family}/{method}: fallback remains {fallback_count}")

        with full_path.open("w", encoding="utf-8") as handle:
            for row in merged:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

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
            + "\n",
            encoding="utf-8",
        )
        print(f"merged {full_path} rows={len(merged)} fallback={fallback_count}")
PY
}

log "waiting for gemma remaining sample completion"
while true; do
    completed="$(gemma_completed_rows)"
    log "gemma_sample_progress=${completed}/473"
    if [[ "${completed}" == "473" ]]; then
        break
    fi
    if ! pgrep -af "src.generate_rubrics.*gemma_small.*rrd_pairwise_sample.*remaining473" >/dev/null; then
        log "ERROR: gemma sample process is not running before completion"
        exit 1
    fi
    sleep 60
done

for attempt in 1 2 3 4 5; do
    ids_file="data/processed/gemma_small_rrd_pairwise_sample_remaining473_finalizer_fallback_attempt${attempt}.ids.txt"
    count="$(fallback_prompt_count "${ids_file}")"
    log "gemma_sample_fallback_after_attempt_$((attempt - 1))=${count}"
    if [[ "${count}" == "0" ]]; then
        break
    fi
    if ! curl -fsS http://127.0.0.1:8111/v1/models >/dev/null; then
        log "ERROR: gemma vLLM not available for fallback repair"
        exit 1
    fi
    OPENAI_BASE_URL="http://127.0.0.1:8111/v1" \
    OPENAI_API_KEY="EMPTY" \
    KCC_RUBRIC_GENERATION_BACKEND="openai_compatible" \
    KCC_EMBED_DEVICE="cpu" \
    python3 -u -m src.generate_rubrics \
        --config configs/rrd_sample_prompt100_gemma_e4b.yaml \
        --generator_family gemma_small \
        --method rrd_pairwise_sample \
        --concurrency 4 \
        --prompt-id-file "${ids_file}" \
        --output-split "remaining473_finalizer_refill${attempt}"
done

merge_full_outputs
log "final merge done; stopping vLLM servers"
pkill -f "vllm.entrypoints.openai.api_server.*Qwen/Qwen3-4B-FP8.*--port 8110" || true
pkill -f "vllm.entrypoints.openai.api_server.*gemma-4-E4B-it.*--port 8111" || true
log "finalizer complete"
