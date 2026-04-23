# CASP: Coupled Action-Set Pessimism

This repository contains the code for the paper:

> CASP: Coupled Action-Set Pessimism for Offline Learning in Two-Stage Recommender Systems

It reproduces the reported simulation study, the reconstructed MovieLens 1M application, and the paper tables and figure inputs. The repository is intentionally focused on the final reported pipeline.

## What Is Included

- `src/casp_sim/`: simulation data-generating processes, learners, estimators, policy library, and runners for Blocks 1--5.
- `src/casp_app/`: MovieLens preprocessing, reconstructed logging model, application environment, comparators, and reports.
- `configs/`: paper configurations for the simulation blocks and MovieLens runs.
- `scripts/`: public entry points for simulations, application runs, robustness checks, and asset building.
- `outputs/`: small reported tables, figure inputs, figures, and manifests.
- `paper/results_map.md`: table-by-table and figure-by-figure reproduction map.

## Setup

Create a Python environment and install the listed dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r environment/requirements.txt
export PYTHONPATH="$PWD/src"
```

## Data

The simulation study generates its own data.

The MovieLens application uses the official MovieLens 1M release. Raw files are not committed. Place the archive or extracted files under `data/raw/` as described in `data/README.md`.

## Reproducing the Main Results

1. Run the simulation suite:

```bash
bash scripts/run_all_simulations.sh
```

2. Run MovieLens preprocessing and the main application:

```bash
bash scripts/run_main_application.sh
```

3. Run the robustness analyses:

```bash
bash scripts/run_movielens_robustness.sh
```

4. Rebuild the reported tables and figure inputs:

```bash
bash scripts/build_paper_outputs.sh
```

The file `paper/results_map.md` shows which script and configuration produce each reported result.

## Quick Repository Guide

- Start with `paper/results_map.md` if you want to reproduce a specific table or figure.
- Use `configs/` to inspect the reported settings directly.
- Use `scripts/run_smoke_tests.sh` for a lightweight verification pass before longer runs.

## Citation and License

If you use this repository, please cite the paper. Citation metadata are provided in `CITATION.cff`.

This repository is released under the MIT License. See `LICENSE`.
