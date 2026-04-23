#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT/src"

python "$ROOT/scripts/build_paper_experiment_assets.py"
python "$ROOT/scripts/build_external_diagnostics.py"
python "$ROOT/scripts/build_application_assets.py"
