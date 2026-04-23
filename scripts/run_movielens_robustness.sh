#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT/src"

python "$ROOT/src/run_movielens_robustness.py" \
  --mode full \
  --output-name tors_robustness_full_v1
