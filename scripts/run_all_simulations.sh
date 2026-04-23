#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT/src"

python "$ROOT/src/run_simulation_suite.py" \
  --config-set phase2 \
  --suite-name phase3_external_baselines_full \
  --max-workers "${CASP_MAX_WORKERS:-0}"

python "$ROOT/src/run_simulation_suite.py" \
  --config-set phase2_block5_precision \
  --suite-name phase3_block5_precision_followup \
  --max-workers "${CASP_MAX_WORKERS:-0}"
