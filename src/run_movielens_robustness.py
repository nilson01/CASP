from __future__ import annotations

import argparse
import csv
import json
import logging
import shutil
from dataclasses import dataclass, replace
from pathlib import Path

from casp_app.config import DEFAULT_APPLICATION_CONFIGS, default_output_root, default_processed_root
from casp_app.pipeline import (
    prepared_dataset_ready,
    run_application_mode,
)
from casp_app.reports import write_rows_csv


BASE_PROCESSED_NAME = "movielens_1m_reconstructed_rebalanced_v2_eps0p10_tau1p00"


@dataclass(frozen=True)
class RobustnessVariant:
    key: str
    label: str
    derivation: str
    overrides: dict
    transform: str
    target_name: str | None = None
    parameter: float | int | None = None


def build_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"casp_app_robustness::{log_path}")
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


def read_summary(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {row["comparator"]: row for row in rows}


def metric(row: dict[str, str], field: str) -> float:
    value = row.get(field, "")
    return float(value) if value else 0.0


def _normalize(values: list[float]) -> list[float]:
    total = sum(values)
    if total <= 1e-12:
        return [1.0 / len(values)] * len(values) if values else []
    return [value / total for value in values]


def _mix_with_uniform(values: list[float], mass: float) -> list[float]:
    if not values:
        return []
    uniform = 1.0 / len(values)
    return _normalize([(1.0 - mass) * value + mass * uniform for value in values])


def _support_generators(candidate_sets: list[list[int]], observed_item: int) -> list[int]:
    return [
        generator
        for generator, candidate_set in enumerate(candidate_sets)
        if observed_item in candidate_set
    ]


def _renormalize_stage1(stage1_probs: list[float], support: list[int], smoothing_mass: float = 0.0) -> list[float]:
    if not support:
        return [0.0] * len(stage1_probs)
    supported_values = [stage1_probs[index] for index in support]
    if sum(supported_values) <= 1e-12:
        supported_values = [1.0 / len(support)] * len(support)
    else:
        supported_values = _normalize(supported_values)
    if smoothing_mass > 0.0:
        supported_values = _mix_with_uniform(supported_values, smoothing_mass)
    out = [0.0] * len(stage1_probs)
    for index, value in zip(support, supported_values):
        out[index] = value
    return out


def _chosen_stage1(stage1_probs: list[float], support: list[int]) -> int:
    return max(support, key=lambda index: (stage1_probs[index], -index))


def _truncate_candidate_sets(row: dict[str, str], size: int) -> dict[str, str] | None:
    candidate_sets = [values[:size] for values in json.loads(row["candidate_sets_json"])]
    stage2_probs = [_normalize(values[:size]) for values in json.loads(row["stage2_probs_json"])]
    observed_item = int(row["observed_item_index"])
    support = _support_generators(candidate_sets, observed_item)
    if not support:
        return None
    stage1_probs = _renormalize_stage1(json.loads(row["stage1_probs_json"]), support)
    row["candidate_sets_json"] = json.dumps(candidate_sets)
    row["stage2_probs_json"] = json.dumps(stage2_probs)
    row["support_generator_count"] = str(len(support))
    row["support_generator_indices_json"] = json.dumps(support)
    row["stage1_probs_json"] = json.dumps(stage1_probs)
    row["stage1_action"] = str(_chosen_stage1(stage1_probs, support))
    return row


def _relabel_reward(row: dict[str, str], threshold: float) -> dict[str, str]:
    row["observed_reward"] = "1" if float(row["observed_rating"]) >= threshold else "0"
    return row


def _smooth_stage1(row: dict[str, str], mass: float) -> dict[str, str]:
    support = json.loads(row["support_generator_indices_json"])
    stage1_probs = _renormalize_stage1(json.loads(row["stage1_probs_json"]), support, smoothing_mass=mass)
    row["stage1_probs_json"] = json.dumps(stage1_probs)
    row["stage1_action"] = str(_chosen_stage1(stage1_probs, support))
    return row


def _smooth_stage2(row: dict[str, str], mass: float) -> dict[str, str]:
    stage2_probs = [_mix_with_uniform(values, mass) for values in json.loads(row["stage2_probs_json"])]
    row["stage2_probs_json"] = json.dumps(stage2_probs)
    return row


def _transform_row(row: dict[str, str], variant: RobustnessVariant) -> dict[str, str] | None:
    if variant.transform == "identity":
        return row
    if variant.transform == "candidate_truncate":
        return _truncate_candidate_sets(row, int(variant.parameter))
    if variant.transform == "reward_relabel":
        return _relabel_reward(row, float(variant.parameter))
    if variant.transform == "stage1_smooth":
        return _smooth_stage1(row, float(variant.parameter))
    if variant.transform == "stage2_smooth":
        return _smooth_stage2(row, float(variant.parameter))
    raise ValueError(f"Unknown robustness transform: {variant.transform}")


def _existing_derivation_manifest(target_processed_root: Path, variant: RobustnessVariant) -> dict | None:
    path = target_processed_root / "derived_variant_manifest.csv"
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return None
    row = dict(rows[0])
    row["status"] = "skipped_existing"
    row.setdefault("variant", variant.key)
    row.setdefault("variant_label", variant.label)
    row.setdefault("derivation", variant.derivation)
    row.setdefault("target_processed_root", str(target_processed_root))
    return row


def derive_processed_dataset(
    base_processed_root: Path,
    target_processed_root: Path,
    variant: RobustnessVariant,
    logger: logging.Logger,
    force: bool = False,
) -> dict:
    if variant.transform == "identity":
        return {
            "variant": variant.key,
            "target_processed_root": str(base_processed_root),
            "status": "uses_base_processed_dataset",
            "source_context_count": "",
            "derived_context_count": "",
            "dropped_context_count": "",
            "dropped_context_share": "",
        }
    if (
        not force
        and (target_processed_root / "catalog.csv").exists()
        and (target_processed_root / "request_contexts.csv").exists()
    ):
        logger.info("Derived data already exists for %s at %s", variant.key, target_processed_root)
        existing = _existing_derivation_manifest(target_processed_root, variant)
        if existing is not None:
            return existing
        return {
            "variant": variant.key,
            "target_processed_root": str(target_processed_root),
            "status": "skipped_existing",
            "variant_label": variant.label,
            "derivation": variant.derivation,
            "source_context_count": "",
            "derived_context_count": "",
            "dropped_context_count": "",
            "dropped_context_share": "",
        }

    target_processed_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(base_processed_root / "catalog.csv", target_processed_root / "catalog.csv")
    source_path = base_processed_root / "request_contexts.csv"
    target_path = target_processed_root / "request_contexts.csv"
    source_count = 0
    kept_count = 0
    with source_path.open(newline="", encoding="utf-8") as source_handle:
        reader = csv.DictReader(source_handle)
        if reader.fieldnames is None:
            raise ValueError(f"Missing header in {source_path}")
        with target_path.open("w", newline="", encoding="utf-8") as target_handle:
            writer = csv.DictWriter(target_handle, fieldnames=reader.fieldnames)
            writer.writeheader()
            for row in reader:
                source_count += 1
                transformed = _transform_row(dict(row), variant)
                if transformed is None:
                    continue
                transformed["request_id"] = str(kept_count)
                writer.writerow(transformed)
                kept_count += 1

    dropped = source_count - kept_count
    manifest = {
        "variant": variant.key,
        "variant_label": variant.label,
        "derivation": variant.derivation,
        "source_processed_root": str(base_processed_root),
        "target_processed_root": str(target_processed_root),
        "source_context_count": source_count,
        "derived_context_count": kept_count,
        "dropped_context_count": dropped,
        "dropped_context_share": dropped / max(source_count, 1),
    }
    write_rows_csv(target_processed_root / "preparation_manifest.csv", [manifest])
    write_rows_csv(target_processed_root / "derived_variant_manifest.csv", [manifest])
    logger.info(
        "Derived %s at %s with %s/%s contexts kept",
        variant.key,
        target_processed_root,
        kept_count,
        source_count,
    )
    return {"status": "derived", **manifest}


def robustness_variants(base_name: str) -> list[RobustnessVariant]:
    return [
        RobustnessVariant(
            key="baseline_cached_L30_ge4",
            label="Baseline cached L=30, rating >=4",
            derivation="accepted prepared dataset; no transformation",
            overrides={},
            transform="identity",
            target_name=base_name,
        ),
        RobustnessVariant(
            key="candidate_L20_refilter_ge4",
            label="Derived L=20 candidate truncation with support refiltering",
            derivation=(
                "cached sensitivity: truncate each accepted top-30 feasible set to top-20, "
                "drop contexts whose observed item loses all generator support, and renormalize supported stage-1 probabilities"
            ),
            overrides={"candidate_set_size": 20},
            transform="candidate_truncate",
            parameter=20,
        ),
        RobustnessVariant(
            key="reward_relabel_L30_ge5",
            label="Reward relabeling rating >=5 on accepted support",
            derivation=(
                "cached sensitivity: keep accepted feasible sets and propensities, but relabel observed rewards using rating >=5"
            ),
            overrides={"positive_rating_threshold": 5.0},
            transform="reward_relabel",
            parameter=5.0,
        ),
        RobustnessVariant(
            key="stage1_smooth020",
            label="Stage-1 logging smoothing +0.20 on accepted support",
            derivation=(
                "cached sensitivity: mix supported stage-1 probabilities with a uniform distribution over supported generators"
            ),
            overrides={"stage1_exploration_epsilon": 0.20},
            transform="stage1_smooth",
            parameter=0.20,
        ),
        RobustnessVariant(
            key="stage2_smooth010",
            label="Stage-2 logging smoothing +0.10 within feasible sets",
            derivation=(
                "cached sensitivity: mix each accepted stage-2 distribution with a uniform distribution over its feasible set"
            ),
            overrides={"min_stage2_mass": 0.10},
            transform="stage2_smooth",
            parameter=0.10,
        ),
    ]


def variant_configs(
    base_config,
    base_name: str,
    mode: str,
    smoke_replications: int,
    full_replications: int,
):
    shared = {
        "stage1_temperature": 1.0,
        "stage1_exploration_epsilon": 0.10,
        "smoke_context_limit": 1800,
        "fallback_context_floor": 3500,
        "fallback_singleton_caps": (6000, 900, 6000, 6000),
    }
    if mode == "smoke":
        shared.update(
            {
                "context_limit": 6000,
                "smoke_replications": smoke_replications,
                "full_replications": max(2, min(full_replications, smoke_replications)),
                "policy_eval_contexts": 1000,
            }
        )
    else:
        shared.update(
            {
                "context_limit": 25000,
                "smoke_replications": smoke_replications,
                "full_replications": full_replications,
                "policy_eval_contexts": 2500,
            }
        )
    for variant in robustness_variants(base_name):
        target_name = variant.target_name or f"{base_name}_torsrobust_{variant.key}"
        config_updates = {**shared, **variant.overrides}
        yield variant, replace(
            base_config,
            name=target_name,
            **config_updates,
        )


def aggregate_variant(output_root: Path, key: str, label: str) -> dict:
    rows = read_summary(output_root / "summary.csv")
    casp = rows["casp_lambda_0.050"]
    dr_only = rows["dr_value_only"]
    ma_style = rows["ma_style_two_stage_opl"]
    return {
        "variant": key,
        "variant_label": label,
        "suite_root": str(output_root),
        "casp_dr_value_mean": metric(casp, "dr_value_mean"),
        "casp_support_burden_mean": metric(casp, "support_burden_mean"),
        "casp_ess_proxy_mean": metric(casp, "ess_proxy_mean"),
        "casp_max_importance_weight_mean": metric(casp, "max_importance_weight_mean"),
        "dr_only_dr_value_mean": metric(dr_only, "dr_value_mean"),
        "dr_only_support_burden_mean": metric(dr_only, "support_burden_mean"),
        "ma_style_dr_value_mean": metric(ma_style, "dr_value_mean"),
        "ma_style_support_burden_mean": metric(ma_style, "support_burden_mean"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run cached MovieLens robustness variants.")
    parser.add_argument("--output-name", default="tors_robustness_smoke")
    parser.add_argument("--base-processed-name", default=BASE_PROCESSED_NAME)
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--smoke-replications", type=int, default=2)
    parser.add_argument("--full-replications", type=int, default=20)
    parser.add_argument("--force-derive", action="store_true")
    parser.add_argument("--force-run", action="store_true")
    args = parser.parse_args()

    parent_root = default_output_root() / args.output_name
    logger = build_logger(parent_root / "logs" / "suite.log")
    base_config = replace(
        DEFAULT_APPLICATION_CONFIGS["movielens_1m_reconstructed"],
        name=args.base_processed_name,
        stage1_temperature=1.0,
        stage1_exploration_epsilon=0.10,
    )
    base_processed_root = default_processed_root() / args.base_processed_name
    if not (base_processed_root / "catalog.csv").exists() or not (base_processed_root / "request_contexts.csv").exists():
        raise FileNotFoundError(
            f"Accepted processed dataset is missing at {base_processed_root}. "
            "Run the accepted prepare-data/rebalanced-v2 pipeline before the fast robustness suite."
        )
    summary_rows = []
    derivation_rows = []

    logger.info(
        "Starting MovieLens robustness %s suite at %s with smoke_replications=%s full_replications=%s",
        args.mode,
        parent_root,
        args.smoke_replications,
        args.full_replications,
    )
    for variant, config in variant_configs(
        base_config,
        args.base_processed_name,
        args.mode,
        args.smoke_replications,
        args.full_replications,
    ):
        variant_root = parent_root / variant.key
        target_processed_root = default_processed_root() / config.name
        logger.info("Variant %s: %s", variant.key, variant.label)
        derivation_rows.append(
            derive_processed_dataset(
                base_processed_root=base_processed_root,
                target_processed_root=target_processed_root,
                variant=variant,
                logger=logger,
                force=args.force_derive,
            )
        )
        if not prepared_dataset_ready(config):
            raise FileNotFoundError(f"Derived processed data are missing for {variant.key}: {target_processed_root}")
        run_application_mode(
            output_root=variant_root,
            config=config,
            mode=args.mode,
            force=args.force_run,
            dry_run=False,
        )
        row = aggregate_variant(variant_root, variant.key, variant.label)
        row["derivation"] = variant.derivation
        row["processed_root"] = str(target_processed_root)
        summary_rows.append(row)
        logger.info("Completed variant %s", variant.key)

    write_rows_csv(parent_root / "robustness_derivation_manifest.csv", derivation_rows)
    write_rows_csv(parent_root / "robustness_summary.csv", summary_rows)
    logger.info("Finished MovieLens robustness %s suite with %s variants", args.mode, len(summary_rows))
    print(f"Robustness artifacts are in {parent_root}")
    print(f"Robustness log: {parent_root / 'logs' / 'suite.log'}")


if __name__ == "__main__":
    main()
