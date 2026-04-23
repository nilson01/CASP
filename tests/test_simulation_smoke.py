from __future__ import annotations

import csv
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from casp_sim.config import DEFAULT_BLOCKS, PHASE2_BLOCKS
from casp_sim.experiments import run_block
from casp_sim.runner import build_logger, run_block_parallel


class SimulationSmokeTests(unittest.TestCase):
    def test_counterexample_runs_and_contains_key_comparators(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_block(DEFAULT_BLOCKS["block1_counterexample"], Path(tmpdir))
            comparators = {row["comparator"] for row in result["rows"]}
            self.assertIn("stagewise_proxy", comparators)
            self.assertIn("dr_value_only", comparators)
            self.assertIn("dr_lcb_beta_1.00", comparators)
            self.assertIn("plugin_reward", comparators)
            self.assertIn("ma_style_two_stage_opl", comparators)
            self.assertIn("wang_style_downstream_generator", comparators)
            self.assertIn("behavior", comparators)
            self.assertTrue(any(name.startswith("casp_lambda_") for name in comparators))

    def test_counterexample_oracle_beats_stagewise_on_average(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_block(DEFAULT_BLOCKS["block1_counterexample"], Path(tmpdir))
            summary = {
                row["comparator"]: row["true_value_mean"]
                for row in result["summary"]
                if row["sweep_value"] == 0.0
            }
            self.assertGreater(summary["oracle"], summary["stagewise_proxy"])

    def test_block_two_is_deterministic(self) -> None:
        config = replace(
            DEFAULT_BLOCKS["block2_coupling"],
            replications=2,
            train_size=120,
            eval_contexts=150,
            sweep_values=(0.0, 0.8),
        )
        with tempfile.TemporaryDirectory() as tmpdir_1, tempfile.TemporaryDirectory() as tmpdir_2:
            result_1 = run_block(config, Path(tmpdir_1))
            result_2 = run_block(config, Path(tmpdir_2))
            self.assertEqual(result_1["summary"], result_2["summary"])

    def test_sample_size_block_runs(self) -> None:
        config = replace(
            DEFAULT_BLOCKS["block5_sample_size"],
            replications=2,
            eval_contexts=100,
            sweep_values=(200.0, 400.0),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_block(config, Path(tmpdir))
            sweep_values = {row["sweep_value"] for row in result["summary"]}
            self.assertEqual(sweep_values, {200.0, 400.0})

    def test_phase2_config_uses_normalized_casp_and_raw_ablation(self) -> None:
        config = replace(
            PHASE2_BLOCKS["block2_coupling"],
            replications=1,
            train_size=120,
            eval_contexts=100,
            sweep_values=(0.5,),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_block(config, Path(tmpdir))
            comparators = {row["comparator"] for row in result["rows"]}
            self.assertIn("casp_ablation_raw_full", comparators)
            casp_rows = [row for row in result["rows"] if row["comparator"].startswith("casp_lambda_")]
            self.assertTrue(all(row.get("selection_penalty_variant") == "library_median" for row in casp_rows))

    def test_parallel_runner_writes_artifacts_and_resumes(self) -> None:
        config = replace(
            DEFAULT_BLOCKS["block1_counterexample"],
            replications=2,
            sweep_values=(0.0,),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            block_dir = Path(tmpdir) / "block1_counterexample"
            logger = build_logger(Path(tmpdir) / "logs" / "suite.log")
            run_block_parallel(config, block_dir, max_workers=2, logger=logger, suite_name="smoke")

            expected_paths = [
                block_dir / "summary.csv",
                block_dir / "selection_frequency.csv",
                block_dir / "plot_manifest.csv",
                block_dir / "task_status.csv",
                block_dir / "progress.csv",
                block_dir / "config.csv",
                block_dir / "block_info.csv",
            ]
            for path in expected_paths:
                self.assertTrue(path.exists(), path)

            task_files = sorted((block_dir / "task_results").glob("*.json"))
            self.assertEqual(len(task_files), 2)

            run_block_parallel(config, block_dir, max_workers=2, logger=logger, suite_name="smoke")
            with (block_dir / "task_status.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertTrue(any(row["status"] == "skipped_completed" for row in rows))

    def test_runner_writes_error_logs_when_task_fails(self) -> None:
        config = replace(
            DEFAULT_BLOCKS["block1_counterexample"],
            replications=1,
            sweep_values=(0.0,),
        )

        def fake_execute(config_dict, sweep_value, replication_index):
            return {
                "status": "error",
                "sweep_value": sweep_value,
                "replication": replication_index,
                "started_at": "2026-01-01T00:00:00+00:00",
                "completed_at": "2026-01-01T00:00:01+00:00",
                "traceback": "synthetic failure",
            }

        with tempfile.TemporaryDirectory() as tmpdir:
            block_dir = Path(tmpdir) / "block1_counterexample"
            logger = build_logger(Path(tmpdir) / "logs" / "suite.log")
            with patch("casp_sim.runner.ProcessPoolExecutor", side_effect=PermissionError("sandbox")):
                with patch("casp_sim.runner.execute_replication_task", side_effect=fake_execute):
                    run_block_parallel(config, block_dir, max_workers=2, logger=logger, suite_name="smoke")

            error_logs = list((block_dir / "errors").glob("*.log"))
            self.assertEqual(len(error_logs), 1)
            self.assertIn("synthetic failure", error_logs[0].read_text(encoding="utf-8"))
            with (block_dir / "task_status.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertTrue(any(row["status"] == "error" for row in rows))


if __name__ == "__main__":
    unittest.main()
