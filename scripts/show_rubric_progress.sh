#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

python_bin="${PYTHON_BIN:-$(command -v python3 || command -v python)}"

"${python_bin}" - <<'PY'
import json
from pathlib import Path

root = Path.cwd() / "data/processed"
progress_files = sorted(root.glob("rubrics_*__*.jsonl.progress.json"))

if not progress_files:
    print("No progress files found.")
    raise SystemExit(0)

for path in progress_files:
    payload = json.loads(path.read_text(encoding="utf-8"))
    done = int(payload.get("completed_rows", 0))
    total = int(payload.get("expected_rows", 0))
    remaining = int(payload.get("remaining_rows", max(total - done, 0)))
    method = payload.get("method", path.name)
    family = payload.get("generator_family", "?")
    pct = (done / total * 100.0) if total else 0.0
    print(f"{family:12s} {method:24s} {done:5d}/{total:<5d} {pct:6.2f}% remaining={remaining}")
PY
