from __future__ import annotations

import csv
import json
import logging
import os
import platform
import socket
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Iterable

from .config import BlockConfig
from .experiments import (
    run_single_replication,
    selection_frequency_rows,
    summarize_rows,
    write_json,
    write_rows_csv,
)
from .plots import generate_block_plots


def timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def recommended_worker_count(task_count: int) -> int:
    cpu_total = os.cpu_count() or 1
    if task_count <= 1:
        return 1
    return max(1, min(task_count, max(1, cpu_total - 1)))


def _scalarize(value) -> str:
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, sort_keys=True)
    return str(value)


def write_single_row_csv(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(payload.keys()))
        writer.writeheader()
        writer.writerow({key: _scalarize(value) for key, value in payload.items()})


def write_atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    temp_path.replace(path)


def task_slug(sweep_name: str, sweep_value: float, replication_index: int) -> str:
    sweep_token = f"{sweep_value:.6g}".replace("-", "neg").replace(".", "p")
    return f"rep_{replication_index:04d}__{sweep_name}_{sweep_token}"


def task_result_path(block_output_dir: Path, sweep_name: str, sweep_value: float, replication_index: int) -> Path:
    return block_output_dir / "task_results" / f"{task_slug(sweep_name, sweep_value, replication_index)}.json"


def task_error_path(block_output_dir: Path, sweep_name: str, sweep_value: float, replication_index: int) -> Path:
    return block_output_dir / "errors" / f"{task_slug(sweep_name, sweep_value, replication_index)}.log"


def enumerate_block_tasks(config: BlockConfig) -> list[tuple[float, int]]:
    return [
        (sweep_value, replication_index)
        for sweep_value in config.sweep_values
        for replication_index in range(config.replications)
    ]


def load_completed_task_payloads(block_output_dir: Path) -> list[dict]:
    task_dir = block_output_dir / "task_results"
    if not task_dir.exists():
        return []
    payloads = []
    for path in sorted(task_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if payload.get("status") == "completed":
            payloads.append(payload)
    return payloads


def aggregate_task_payloads(block_output_dir: Path, config: BlockConfig) -> dict:
    payloads = load_completed_task_payloads(block_output_dir)
    rows = []
    for payload in payloads:
        rows.extend(payload.get("rows", []))
    summary_rows = summarize_rows(rows)
    selection_rows = selection_frequency_rows(rows)
    plot_manifest = generate_block_plots(block_output_dir, summary_rows)

    write_rows_csv(block_output_dir / "per_replication.csv", rows)
    write_rows_csv(block_output_dir / "summary.csv", summary_rows)
    write_rows_csv(block_output_dir / "selection_frequency.csv", selection_rows)
    write_json(block_output_dir / "summary.json", summary_rows)
    write_json(block_output_dir / "selection_frequency.json", selection_rows)
    write_json(block_output_dir / "plot_manifest.json", plot_manifest)
    write_rows_csv(block_output_dir / "plot_manifest.csv", plot_manifest)
    write_json(block_output_dir / "config.json", config.to_dict())
    write_single_row_csv(block_output_dir / "config.csv", config.to_dict())
    return {
        "rows": rows,
        "summary": summary_rows,
        "selection_frequency": selection_rows,
        "plot_manifest": plot_manifest,
        "completed_tasks": len(payloads),
    }


def execute_replication_task(config_dict: dict, sweep_value: float, replication_index: int) -> dict:
    started_at = timestamp()
    try:
        config = BlockConfig(**config_dict)
        rows = run_single_replication(config, sweep_value, replication_index)
        completed_at = timestamp()
        return {
            "status": "completed",
            "block": config.name,
            "sweep_name": config.sweep_name,
            "sweep_value": sweep_value,
            "replication": replication_index,
            "started_at": started_at,
            "completed_at": completed_at,
            "rows": rows,
        }
    except Exception:
        completed_at = timestamp()
        return {
            "status": "error",
            "sweep_value": sweep_value,
            "replication": replication_index,
            "started_at": started_at,
            "completed_at": completed_at,
            "traceback": traceback.format_exc(),
        }


def suite_metadata_dict(suite_name: str, blocks: Iterable[str], max_workers: int) -> dict:
    return {
        "suite_name": suite_name,
        "created_at": timestamp(),
        "working_directory": str(Path.cwd()),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": sys.version.split()[0],
        "cpu_count": os.cpu_count() or 1,
        "max_workers": max_workers,
        "blocks": list(blocks),
    }


def build_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger(f"casp_runner_{log_path}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def run_block_parallel(
    config: BlockConfig,
    block_output_dir: Path,
    max_workers: int,
    logger: logging.Logger,
    suite_name: str,
    force: bool = False,
) -> dict:
    block_output_dir.mkdir(parents=True, exist_ok=True)
    all_tasks = enumerate_block_tasks(config)
    status_rows = []
    progress_rows = []
    pending = []
    for sweep_value, replication_index in all_tasks:
        result_path = task_result_path(block_output_dir, config.sweep_name, sweep_value, replication_index)
        if result_path.exists() and not force:
            status_rows.append(
                {
                    "block": config.name,
                    "suite_name": suite_name,
                    "sweep_value": sweep_value,
                    "replication": replication_index,
                    "status": "skipped_completed",
                    "started_at": "",
                    "completed_at": timestamp(),
                    "task_file": str(result_path),
                    "error_file": "",
                }
            )
            continue
        pending.append((sweep_value, replication_index, result_path))

    logger.info("Block %s: %s total tasks, %s pending, %s already completed", config.name, len(all_tasks), len(pending), len(all_tasks) - len(pending))
    for sweep_value, replication_index, result_path in pending:
        progress_rows.append(
            {
                "timestamp": timestamp(),
                "block": config.name,
                "event": "queued",
                "sweep_value": sweep_value,
                "replication": replication_index,
                "message": str(result_path),
            }
        )

    def record_payload(payload: dict, sweep_value: float, replication_index: int, result_path: Path) -> None:
        error_path = task_error_path(block_output_dir, config.sweep_name, sweep_value, replication_index)
        if payload["status"] == "completed":
            write_atomic_json(result_path, payload)
            logger.info(
                "Completed block=%s sweep=%s replication=%s",
                config.name,
                sweep_value,
                replication_index,
            )
            status_rows.append(
                {
                    "block": config.name,
                    "suite_name": suite_name,
                    "sweep_value": sweep_value,
                    "replication": replication_index,
                    "status": "completed",
                    "started_at": payload["started_at"],
                    "completed_at": payload["completed_at"],
                    "task_file": str(result_path),
                    "error_file": "",
                }
            )
            progress_rows.append(
                {
                    "timestamp": timestamp(),
                    "block": config.name,
                    "event": "completed",
                    "sweep_value": sweep_value,
                    "replication": replication_index,
                    "message": str(result_path),
                }
            )
            return

        error_path.parent.mkdir(parents=True, exist_ok=True)
        error_path.write_text(payload.get("traceback", "Unknown error"), encoding="utf-8")
        logger.error(
            "Failed block=%s sweep=%s replication=%s -> %s",
            config.name,
            sweep_value,
            replication_index,
            error_path,
        )
        status_rows.append(
            {
                "block": config.name,
                "suite_name": suite_name,
                "sweep_value": sweep_value,
                "replication": replication_index,
                "status": "error",
                "started_at": payload.get("started_at", ""),
                "completed_at": payload.get("completed_at", ""),
                "task_file": str(result_path),
                "error_file": str(error_path),
            }
        )
        progress_rows.append(
            {
                "timestamp": timestamp(),
                "block": config.name,
                "event": "error",
                "sweep_value": sweep_value,
                "replication": replication_index,
                "message": str(error_path),
            }
        )

    if pending:
        worker_count = max(1, min(max_workers, len(pending)))
        try:
            with ProcessPoolExecutor(max_workers=worker_count) as executor:
                future_map = {
                    executor.submit(execute_replication_task, config.to_dict(), sweep_value, replication_index): (
                        sweep_value,
                        replication_index,
                        result_path,
                    )
                    for sweep_value, replication_index, result_path in pending
                }
                for future in as_completed(future_map):
                    sweep_value, replication_index, result_path = future_map[future]
                    try:
                        payload = future.result()
                    except Exception:
                        payload = {
                            "status": "error",
                            "sweep_value": sweep_value,
                            "replication": replication_index,
                            "started_at": "",
                            "completed_at": timestamp(),
                            "traceback": traceback.format_exc(),
                        }
                    record_payload(payload, sweep_value, replication_index, result_path)
        except (PermissionError, OSError) as error:
            logger.warning(
                "Falling back to sequential execution for block %s because process-based parallelism is unavailable: %s",
                config.name,
                error,
            )
            for sweep_value, replication_index, result_path in pending:
                payload = execute_replication_task(config.to_dict(), sweep_value, replication_index)
                record_payload(payload, sweep_value, replication_index, result_path)

    aggregate = aggregate_task_payloads(block_output_dir, config)
    status_rows.sort(key=lambda row: (row["sweep_value"], row["replication"], row["status"]))
    progress_rows.sort(key=lambda row: (row["timestamp"], row["block"], row["replication"]))
    write_rows_csv(block_output_dir / "task_status.csv", status_rows)
    write_json(block_output_dir / "task_status.json", status_rows)
    write_rows_csv(block_output_dir / "progress.csv", progress_rows)
    block_info = {
        "block": config.name,
        "updated_at": timestamp(),
        "total_tasks": len(all_tasks),
        "completed_tasks": aggregate["completed_tasks"],
        "failed_tasks": sum(1 for row in status_rows if row["status"] == "error"),
        "pending_tasks": len(all_tasks) - aggregate["completed_tasks"] - sum(1 for row in status_rows if row["status"] == "error"),
        "suite_name": suite_name,
    }
    write_json(block_output_dir / "block_info.json", block_info)
    write_single_row_csv(block_output_dir / "block_info.csv", block_info)
    return {
        "config": config.to_dict(),
        "summary": aggregate["summary"],
        "selection_frequency": aggregate["selection_frequency"],
        "plot_manifest": aggregate["plot_manifest"],
        "status": status_rows,
        "block_info": block_info,
    }
