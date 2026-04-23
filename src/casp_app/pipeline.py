from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from time import perf_counter

from .comparators import application_comparators
from .config import ApplicationConfig, RunModeSpec, default_processed_root, default_run_mode_specs, run_mode_spec
from .datasets import (
    dataset_manifest_rows,
    load_movies,
    load_ratings,
    load_users,
    provenance_rows,
    raw_data_available,
    require_raw_files,
)
from .experiment import (
    aggregate_logging_rows,
    frontier_rows,
    run_single_application_split,
    selection_frequency_rows,
    summarize_application_rows,
)
from .reconstruction import generator_manifest_rows, logging_design_rows, prepare_reconstructed_dataset, reconstruction_plan_rows
from .reports import comparator_manifest_rows, figure_table_map, output_contract_rows, write_json, write_rows_csv


def _timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _single_row_csv(path: Path, payload: dict) -> None:
    write_rows_csv(path, [payload])


def _build_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"casp_app::{log_path}")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def prepared_dataset_root(config: ApplicationConfig) -> Path:
    return default_processed_root() / config.name


def prepared_dataset_ready(config: ApplicationConfig) -> bool:
    root = prepared_dataset_root(config)
    return (root / "catalog.csv").exists() and (root / "request_contexts.csv").exists()


def _write_foundation_artifacts(output_root: Path, config: ApplicationConfig) -> None:
    write_json(output_root / "application_config.json", config.to_dict())
    _single_row_csv(output_root / "application_config.csv", config.to_dict())
    write_rows_csv(output_root / "dataset_manifest.csv", dataset_manifest_rows(config))
    write_json(output_root / "dataset_manifest.json", dataset_manifest_rows(config))
    write_rows_csv(output_root / "dataset_provenance.csv", provenance_rows(config))
    write_json(output_root / "dataset_provenance.json", provenance_rows(config))
    write_rows_csv(output_root / "generator_manifest.csv", generator_manifest_rows(config))
    write_json(output_root / "generator_manifest.json", generator_manifest_rows(config))
    write_rows_csv(output_root / "logging_design.csv", logging_design_rows(config))
    write_json(output_root / "logging_design.json", logging_design_rows(config))
    write_rows_csv(output_root / "reconstruction_plan.csv", reconstruction_plan_rows(config))
    write_json(output_root / "reconstruction_plan.json", reconstruction_plan_rows(config))
    write_rows_csv(output_root / "comparator_manifest.csv", comparator_manifest_rows())
    write_json(output_root / "comparator_manifest.json", comparator_manifest_rows())
    write_rows_csv(output_root / "figure_table_map.csv", figure_table_map(config))
    write_json(output_root / "figure_table_map.json", figure_table_map(config))
    write_rows_csv(output_root / "output_contract.csv", output_contract_rows(config))
    write_json(output_root / "output_contract.json", output_contract_rows(config))
    run_specs = [spec.to_dict() for spec in default_run_mode_specs(config).values()]
    write_rows_csv(output_root / "run_modes.csv", run_specs)
    write_json(output_root / "run_modes.json", run_specs)


def build_application_foundation(
    output_root: Path,
    config: ApplicationConfig,
) -> dict:
    output_root.mkdir(parents=True, exist_ok=True)
    suite_info = {
        "suite_name": output_root.name,
        "created_at": _timestamp(),
        "dataset": config.name,
        "status": "foundation_ready",
        "raw_data_available": raw_data_available(config),
        "prepared_dataset_ready": prepared_dataset_ready(config),
    }
    write_json(output_root / "suite_info.json", suite_info)
    _single_row_csv(output_root / "suite_info.csv", suite_info)
    _write_foundation_artifacts(output_root, config)
    foundation_note = {
        "dataset_choice": "MovieLens 1M reconstructed two-stage logger",
        "generator_count": len(config.generators),
        "candidate_set_size": config.candidate_set_size,
        "comparator_count": len(application_comparators()),
        "next_step": "Use validate mode for static checks, then prepare-data once the raw files are placed.",
    }
    write_json(output_root / "foundation_note.json", foundation_note)
    _single_row_csv(output_root / "foundation_note.csv", foundation_note)
    return suite_info


def prepare_application_data(output_root: Path, config: ApplicationConfig) -> dict:
    require_raw_files(config, "prepare-data")
    output_root.mkdir(parents=True, exist_ok=True)
    logger = _build_logger(output_root / "logs" / "suite.log")
    logger.info("Preparing reconstructed application data for %s", config.name)
    users = load_users(config)
    movies = load_movies(config)
    ratings = load_ratings(config)
    logger.info(
        "Loaded raw MovieLens tables: users=%s movies=%s ratings=%s",
        len(users),
        len(movies),
        len(ratings),
    )
    processed_root = prepared_dataset_root(config)
    manifest = prepare_reconstructed_dataset(
        config=config,
        users=users,
        movies=movies,
        ratings=ratings,
        processed_root=processed_root,
        logger=logger,
    )
    suite_info = {
        "suite_name": output_root.name,
        "created_at": _timestamp(),
        "dataset": config.name,
        "status": "prepared_data_ready",
        "processed_root": str(processed_root),
        **manifest,
    }
    write_json(output_root / "suite_info.json", suite_info)
    _single_row_csv(output_root / "suite_info.csv", suite_info)
    _write_foundation_artifacts(output_root, config)
    logger.info("Prepared reconstructed data at %s", processed_root)
    return suite_info


def validate_application_mode(output_root: Path, config: ApplicationConfig, mode: str) -> dict:
    output_root.mkdir(parents=True, exist_ok=True)
    run_spec = run_mode_spec(config, mode)
    suite_info = {
        "suite_name": output_root.name,
        "created_at": _timestamp(),
        "dataset": config.name,
        "mode": mode,
        "status": f"validated_{mode}",
        "run_spec": run_spec.to_dict(),
        "raw_data_available": raw_data_available(config),
        "prepared_dataset_ready": prepared_dataset_ready(config),
    }
    write_json(output_root / "suite_info.json", suite_info)
    _single_row_csv(output_root / "suite_info.csv", suite_info)
    _write_foundation_artifacts(output_root, config)
    return suite_info


def _load_existing_task_results(task_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(task_dir.glob("replication_*.json")):
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    return rows


def _write_application_outputs(output_root: Path, task_payloads: list[dict], run_spec: RunModeSpec) -> None:
    all_result_rows = []
    generator_delta_rows = []
    item_delta_rows = []
    logging_rows_by_replication = []
    for payload in task_payloads:
        all_result_rows.extend(payload["result_rows"])
        generator_delta_rows.extend(payload["generator_delta_rows"])
        item_delta_rows.extend(payload["item_delta_rows"])
        logging_rows_by_replication.append(payload["logging_rows"])

    summary_rows = summarize_application_rows(all_result_rows)
    selection_rows = selection_frequency_rows(all_result_rows)
    frontier = frontier_rows(summary_rows)
    logging_rows = aggregate_logging_rows(logging_rows_by_replication)

    write_rows_csv(output_root / "summary.csv", summary_rows)
    write_rows_csv(output_root / "summary_full.csv", all_result_rows)
    write_rows_csv(output_root / "selection_frequency.csv", selection_rows)
    write_rows_csv(output_root / "policy_delta_generators.csv", generator_delta_rows)
    write_rows_csv(output_root / "policy_delta_items.csv", item_delta_rows)
    write_rows_csv(output_root / "logging_diagnostics.csv", logging_rows)
    write_rows_csv(output_root / "diagnostics" / "value_burden_frontier.csv", frontier)

    run_manifest = {
        "mode": run_spec.mode,
        "context_cap": run_spec.context_cap,
        "split_replications": run_spec.split_replications,
        "holdout_fraction": run_spec.holdout_fraction,
        "policy_eval_contexts": run_spec.policy_eval_contexts,
        "include_ablations": run_spec.include_ablations,
    }
    write_json(output_root / "run_manifest.json", run_manifest)
    _single_row_csv(output_root / "run_manifest.csv", run_manifest)


def _write_progress(output_root: Path, task_rows: list[dict], total_tasks: int) -> dict:
    completed = sum(1 for row in task_rows if row["status"] == "completed")
    failed = sum(1 for row in task_rows if row["status"] == "error")
    skipped = sum(1 for row in task_rows if row["status"] == "skipped_existing")
    attempted = len(task_rows)
    finished = completed + skipped
    progress_row = {
        "completed_tasks": completed,
        "failed_tasks": failed,
        "skipped_existing": skipped,
        "attempted_tasks": attempted,
        "finished_tasks": finished,
        "total_tasks": total_tasks,
        "completion_fraction": finished / max(total_tasks, 1),
        "completion_percent": 100.0 * finished / max(total_tasks, 1),
        "last_updated_at": _timestamp(),
    }
    _single_row_csv(output_root / "progress.csv", progress_row)
    return progress_row


def run_application_mode(
    output_root: Path,
    config: ApplicationConfig,
    mode: str,
    force: bool = False,
    dry_run: bool = False,
) -> dict:
    run_spec = run_mode_spec(config, mode)
    output_root.mkdir(parents=True, exist_ok=True)
    logger = _build_logger(output_root / "logs" / "suite.log")
    suite_info = {
        "suite_name": output_root.name,
        "created_at": _timestamp(),
        "dataset": config.name,
        "mode": mode,
        "force_rerun": force,
        "dry_run": dry_run,
        "status": "running",
        "run_spec": run_spec.to_dict(),
    }
    write_json(output_root / "suite_info.json", suite_info)
    _single_row_csv(output_root / "suite_info.csv", suite_info)
    _write_foundation_artifacts(output_root, config)

    if dry_run:
        suite_info["status"] = f"validated_{mode}"
        suite_info["finished_at"] = _timestamp()
        write_json(output_root / "suite_info.json", suite_info)
        _single_row_csv(output_root / "suite_info.csv", suite_info)
        logger.info("Validated application mode %s without touching data", mode)
        return suite_info

    if not prepared_dataset_ready(config):
        if not raw_data_available(config):
            require_raw_files(config, mode)
        raise FileNotFoundError(
            f"Prepared application data are missing for mode '{mode}'. "
            f"Run prepare-data first so that {prepared_dataset_root(config)} contains catalog.csv and request_contexts.csv."
        )

    logger.info("Starting application mode %s with replications=%s force=%s", mode, run_spec.split_replications, force)
    task_dir = output_root / "task_results"
    error_dir = output_root / "errors"
    task_dir.mkdir(parents=True, exist_ok=True)
    error_dir.mkdir(parents=True, exist_ok=True)

    task_rows = []
    _write_progress(output_root, task_rows, run_spec.split_replications)
    for replication_index in range(run_spec.split_replications):
        replication_start = perf_counter()
        task_path = task_dir / f"replication_{replication_index:03d}.json"
        error_path = error_dir / f"replication_{replication_index:03d}.log"
        logger.info(
            "Starting replication %s/%s",
            replication_index + 1,
            run_spec.split_replications,
        )
        if task_path.exists() and not force:
            task_rows.append({"replication": replication_index, "status": "skipped_existing", "task_file": str(task_path), "error_file": ""})
            progress_row = _write_progress(output_root, task_rows, run_spec.split_replications)
            logger.info(
                "Skipped existing replication %s; completed=%s/%s (%.1f%%)",
                replication_index,
                progress_row["completed_tasks"],
                progress_row["total_tasks"],
                progress_row["completion_percent"],
            )
            continue
        try:
            payload = run_single_application_split(
                processed_root=prepared_dataset_root(config),
                config=config,
                run_spec=run_spec,
                replication_index=replication_index,
            )
            task_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
            if error_path.exists():
                error_path.unlink()
            task_rows.append({"replication": replication_index, "status": "completed", "task_file": str(task_path), "error_file": ""})
            progress_row = _write_progress(output_root, task_rows, run_spec.split_replications)
            logger.info(
                "Completed replication %s in %.1fs; completed=%s/%s (%.1f%%)",
                replication_index,
                perf_counter() - replication_start,
                progress_row["completed_tasks"],
                progress_row["total_tasks"],
                progress_row["completion_percent"],
            )
        except Exception as error:  # pragma: no cover - error path is for resumability
            error_path.write_text(f"{type(error).__name__}: {error}\n", encoding="utf-8")
            task_rows.append({"replication": replication_index, "status": "error", "task_file": str(task_path), "error_file": str(error_path)})
            logger.error("Replication %s failed: %s", replication_index, error)
            _write_progress(output_root, task_rows, run_spec.split_replications)

    write_rows_csv(output_root / "task_status.csv", task_rows)
    progress_row = _write_progress(output_root, task_rows, run_spec.split_replications)

    if progress_row["failed_tasks"] == 0:
        payloads = _load_existing_task_results(task_dir)
        _write_application_outputs(output_root, payloads, run_spec)

    suite_info["finished_at"] = _timestamp()
    suite_info["status"] = "completed_with_errors" if progress_row["failed_tasks"] else "completed"
    suite_info["progress"] = progress_row
    write_json(output_root / "suite_info.json", suite_info)
    _single_row_csv(output_root / "suite_info.csv", suite_info)
    logger.info("Finished application mode %s with status=%s", mode, suite_info["status"])
    return suite_info
