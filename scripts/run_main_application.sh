#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT/src"

python "$ROOT/src/run_application.py" \
  --mode prepare-data \
  --dataset movielens_1m_reconstructed \
  --suite-name prepare_movielens_1m_generator2_support_v2_fix1 \
  --config-name-override movielens_1m_reconstructed_generator2_support_v2_fix1 \
  --stage1-exploration-epsilon 0.10 \
  --stage1-temperature 1.00

python "$ROOT/src/run_application.py" \
  --mode full \
  --dataset movielens_1m_reconstructed \
  --suite-name tors_full_movielens_1m_rebalanced_v2_eps0p10_tau1p00 \
  --config-name-override movielens_1m_reconstructed_generator2_support_v2_fix1 \
  --stage1-exploration-epsilon 0.10 \
  --stage1-temperature 1.00
