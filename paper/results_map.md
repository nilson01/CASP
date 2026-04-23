# Paper Results Map

This file maps the reported manuscript results to the scripts, configurations, and output assets in the repository.

## Main Simulation Results

### Block 1 counterexample table

- Script: `src/run_simulation_suite.py`
- Config: `configs/simulation/block1.yaml`
- Command: `bash scripts/run_all_simulations.sh`
- Manuscript asset: `outputs/tables/simulation/block1_counterexample_table.tex`
- Built asset: `outputs/paper_assets/simulation/phase3_external_baselines/block1_counterexample_table.tex`
- Run directory: `outputs/runs/simulation/phase3_external_baselines_full/block1_counterexample/`

### Block 2 coupling sweep figure

- Script: `src/run_simulation_suite.py`
- Config: `configs/simulation/block2.yaml`
- Command: `bash scripts/run_all_simulations.sh`
- Manuscript figure input: `outputs/figures/simulation/block2_coupling_key.csv`
- Built figure input: `outputs/paper_assets/simulation/phase3_external_baselines/block2_coupling_key.csv`
- Run directory: `outputs/runs/simulation/phase3_external_baselines_full/block2_coupling/`

### Blocks 2--5 cross-block frontier

- Script: `src/run_simulation_suite.py`
- Configs: `configs/simulation/block2.yaml`, `configs/simulation/block3.yaml`, `configs/simulation/block4.yaml`, `configs/simulation/block5.yaml`
- Command: `bash scripts/run_all_simulations.sh`
- Manuscript table: `outputs/tables/simulation/crossblock_summary_table.tex`
- Manuscript figure input: `outputs/figures/simulation/simulation_frontier_key.csv`
- Built assets: `outputs/paper_assets/simulation/phase3_external_baselines/`
- Run directory: `outputs/runs/simulation/phase3_external_baselines_full/`

### Block 5 precision addendum

- Script: `src/run_simulation_suite.py`
- Config: `configs/simulation/block5_precision.yaml`
- Command: `bash scripts/run_all_simulations.sh`
- Manuscript asset: `outputs/tables/simulation/block5_precision_summary.csv`
- Built assets: `outputs/paper_assets/simulation/phase3_block5_precision_followup/`
- Run directory: `outputs/runs/simulation/phase3_block5_precision_followup/block5_sample_size/`

## MovieLens Application Results

### Main comparator table

- Script: `src/run_application.py`
- Config: `configs/movielens/main.yaml`
- Command: `bash scripts/run_main_application.sh`
- Manuscript asset: `outputs/tables/application/application_main_comparator_table.tex`
- Built asset: `outputs/paper_assets/application/application_main_comparator_table.tex`
- Run directory: `outputs/runs/application/tors_full_movielens_1m_rebalanced_v2_eps0p10_tau1p00/`

### Application value--burden frontier

- Script: `src/run_application.py`
- Config: `configs/movielens/main.yaml`
- Command: `bash scripts/run_main_application.sh`
- Manuscript figure input: `outputs/figures/application/application_frontier_key.csv`
- Built figure input: `outputs/paper_assets/application/frontier_key.csv`

### Generator-level policy delta

- Script: `src/run_application.py`
- Config: `configs/movielens/main.yaml`
- Command: `bash scripts/run_main_application.sh`
- Manuscript figure input: `outputs/figures/application/policy_delta_generator_shares.csv`
- Built figure input: `outputs/paper_assets/application/policy_delta_generator_shares.csv`

### Lambda sensitivity and support diagnostics

- Script: `src/run_application.py`
- Config: `configs/movielens/main.yaml`
- Command: `bash scripts/run_main_application.sh`
- Manuscript lambda table input: `outputs/tables/application/lambda_sensitivity.csv`
- Manuscript support diagnostic input: `outputs/tables/application/support_violation_diagnostics.csv`

### MovieLens robustness table

- Script: `src/run_movielens_robustness.py`
- Config: `configs/movielens/robustness.yaml`
- Command: `bash scripts/run_movielens_robustness.sh`
- Manuscript asset: `outputs/tables/application/appendix_application_robustness_table.tex`
- Built asset: `outputs/paper_assets/application/appendix_application_robustness_table.tex`
- Run directory: `outputs/runs/application/tors_robustness_full_v1/`

## Asset Builders

- Simulation tables and figure inputs: `scripts/build_paper_experiment_assets.py`
- Simulation diagnostics and figure inputs: `scripts/build_external_diagnostics.py`
- Application tables and figure inputs: `scripts/build_application_assets.py`
- Wrapper command: `bash scripts/build_paper_outputs.sh`
