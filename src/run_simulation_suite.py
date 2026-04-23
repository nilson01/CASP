from __future__ import annotations

import argparse
from pathlib import Path

from casp_sim.config import BLOCK_SETS, DEFAULT_BLOCKS, default_output_root, locked_sweep_grid_rows, materialize_figure_table_map
from casp_sim.experiments import write_json, write_rows_csv
from casp_sim.runner import build_logger, recommended_worker_count, run_block_parallel, suite_metadata_dict, timestamp, write_single_row_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the CASP simulation suite.")
    parser.add_argument(
        "--block",
        default="all",
        choices=["all", *DEFAULT_BLOCKS.keys()],
        help="Which block to run.",
    )
    parser.add_argument(
        "--config-set",
        default="phase1",
        choices=sorted(BLOCK_SETS.keys()),
        help="Which locked block/config set to run.",
    )
    parser.add_argument(
        "--suite-name",
        default="phase3_default",
        help="Output subdirectory name under outputs/runs/simulation.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=0,
        help="Number of worker processes. Use 0 for an automatic CPU-based choice.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun tasks even if completed task results already exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    blocks = BLOCK_SETS[args.config_set]
    output_root = default_output_root() / args.suite_name
    output_root.mkdir(parents=True, exist_ok=True)

    block_names = list(blocks.keys()) if args.block == "all" else [args.block]
    total_tasks = sum(
        blocks[block_name].replications * len(blocks[block_name].sweep_values)
        for block_name in block_names
    )
    max_workers = args.max_workers if args.max_workers > 0 else recommended_worker_count(total_tasks)
    suite_metadata = suite_metadata_dict(args.suite_name, block_names, max_workers=max_workers)
    suite_metadata["output_root"] = str(output_root)
    suite_metadata["requested_block"] = args.block
    suite_metadata["config_set"] = args.config_set
    suite_metadata["force_rerun"] = args.force
    write_json(output_root / "suite_info.json", suite_metadata)
    write_single_row_csv(output_root / "suite_info.csv", suite_metadata)
    sweep_grid = locked_sweep_grid_rows(blocks)
    write_json(output_root / "locked_sweep_grid.json", sweep_grid)
    write_rows_csv(output_root / "locked_sweep_grid.csv", sweep_grid)
    figure_table_map = materialize_figure_table_map(output_root)
    write_json(output_root / "figure_table_map.json", figure_table_map)
    write_rows_csv(output_root / "figure_table_map.csv", figure_table_map)
    logger = build_logger(output_root / "logs" / "suite.log")
    logger.info("Starting suite %s with config_set=%s blocks=%s max_workers=%s force=%s", args.suite_name, args.config_set, block_names, max_workers, args.force)

    manifest = {}
    manifest_rows = []
    for block_name in block_names:
        config = blocks[block_name]
        block_output = output_root / block_name
        print(f"Running {block_name} -> {block_output} with max_workers={max_workers}")
        result = run_block_parallel(
            config,
            block_output,
            max_workers=max_workers,
            logger=logger,
            suite_name=args.suite_name,
            force=args.force,
        )
        manifest[block_name] = {
            "config": result["config"],
            "summary_file": str(block_output / "summary.json"),
            "summary_table_file": str(block_output / "summary.csv"),
            "replication_file": str(block_output / "per_replication.csv"),
            "selection_frequency_file": str(block_output / "selection_frequency.json"),
            "selection_frequency_table_file": str(block_output / "selection_frequency.csv"),
            "plot_manifest_file": str(block_output / "plot_manifest.json"),
            "plot_manifest_table_file": str(block_output / "plot_manifest.csv"),
            "task_status_file": str(block_output / "task_status.csv"),
            "progress_file": str(block_output / "progress.csv"),
        }
        manifest_rows.append(
            {
                "block": block_name,
                "summary_table_file": str(block_output / "summary.csv"),
                "replication_file": str(block_output / "per_replication.csv"),
                "selection_frequency_table_file": str(block_output / "selection_frequency.csv"),
                "plot_manifest_table_file": str(block_output / "plot_manifest.csv"),
                "task_status_file": str(block_output / "task_status.csv"),
                "progress_file": str(block_output / "progress.csv"),
                "completed_tasks": result["block_info"]["completed_tasks"],
                "failed_tasks": result["block_info"]["failed_tasks"],
                "pending_tasks": result["block_info"]["pending_tasks"],
            }
        )

    write_json(output_root / "manifest.json", manifest)
    if manifest_rows:
        write_rows_csv(output_root / "manifest.csv", manifest_rows)
    suite_metadata["finished_at"] = timestamp()
    suite_metadata["status"] = "completed_with_errors" if any(row["failed_tasks"] > 0 for row in manifest_rows) else "completed"
    write_json(output_root / "suite_info.json", suite_metadata)
    write_single_row_csv(output_root / "suite_info.csv", suite_metadata)
    logger.info("Finished suite %s", args.suite_name)
    print(f"Finished. Artifacts are in {output_root}")


if __name__ == "__main__":
    main()
