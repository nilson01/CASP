from __future__ import annotations

import csv
import json
from pathlib import Path

from .comparators import application_comparators
from .config import ApplicationConfig, FigureTableSpec


def write_rows_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen_fields: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen_fields:
                seen_fields.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def figure_table_map(config: ApplicationConfig) -> list[dict]:
    specs = [
        FigureTableSpec(
            asset_id="app_dataset_summary_table",
            asset_type="table",
            manuscript_location="main_text.application",
            title="Application dataset summary",
            source_path_hint="dataset_manifest.csv",
            note="Counts, warm-start rule, and logging reconstruction summary.",
        ),
        FigureTableSpec(
            asset_id="app_run_manifest_table",
            asset_type="table",
            manuscript_location="appendix.application",
            title="Application run manifest",
            source_path_hint="run_manifest.csv",
            note="Smoke/full configuration, holdout fraction, and replication count.",
        ),
        FigureTableSpec(
            asset_id="app_generator_library_table",
            asset_type="table",
            manuscript_location="main_text.application",
            title="Generator library and support interpretation",
            source_path_hint="generator_manifest.csv",
            note="Explains the finite stage-1 library used in the application.",
        ),
        FigureTableSpec(
            asset_id="app_value_burden_table",
            asset_type="table",
            manuscript_location="main_text.application",
            title="Application comparator summary",
            source_path_hint="summary.csv",
            note="Main reported comparator table with value, burden, and stability.",
        ),
        FigureTableSpec(
            asset_id="app_support_diagnostic_figure",
            asset_type="figure",
            manuscript_location="main_text.application",
            title="Application value-burden diagnostic",
            source_path_hint="diagnostics/value_burden_frontier.csv",
            note="Main reported support diagnostic figure.",
        ),
        FigureTableSpec(
            asset_id="app_policy_delta_generator_table",
            asset_type="table",
            manuscript_location="main_text.application",
            title="Generator-level policy delta",
            source_path_hint="policy_delta_generators.csv",
            note="Shows where CASP and DR-value-only prefer different generators.",
        ),
        FigureTableSpec(
            asset_id="app_policy_delta_item_table",
            asset_type="table",
            manuscript_location="appendix.application",
            title="Item-level policy delta",
            source_path_hint="policy_delta_items.csv",
            note="Highlights feasible-set or item-level shifts under weak support.",
        ),
        FigureTableSpec(
            asset_id="app_appendix_full_comparator_table",
            asset_type="table",
            manuscript_location="appendix.application",
            title="Full comparator table",
            source_path_hint="summary_full.csv",
            note="Retains the full comparator layer used in the application.",
        ),
        FigureTableSpec(
            asset_id="app_appendix_logging_table",
            asset_type="table",
            manuscript_location="appendix.application",
            title="Logging and support diagnostics",
            source_path_hint="logging_diagnostics.csv",
            note="Retains logger concentration, feasible-support coverage, and support-deficiency summaries.",
        ),
        FigureTableSpec(
            asset_id="app_appendix_ablation_table",
            asset_type="table",
            manuscript_location="appendix.application",
            title="Application ablation summary",
            source_path_hint="summary.csv",
            note="Ablation rows at lambda 0.05 for normalized full, stage-1 only, stage-2 only, and raw full.",
        ),
        FigureTableSpec(
            asset_id="app_appendix_lambda_sensitivity_table",
            asset_type="table",
            manuscript_location="appendix.application",
            title="Application lambda sensitivity",
            source_path_hint="figures/application_assets/lambda_sensitivity.csv",
            note="Calibration check over the CASP penalty grid.",
        ),
        FigureTableSpec(
            asset_id="app_appendix_weight_diagnostics_table",
            asset_type="table",
            manuscript_location="appendix.application",
            title="Effective sample size and max-weight diagnostics",
            source_path_hint="figures/application_assets/appendix_application_weight_diagnostics_table.tex",
            note="Documents whether support-burden reductions also reduce tail-weight risk.",
        ),
        FigureTableSpec(
            asset_id="app_appendix_robustness_table",
            asset_type="table",
            manuscript_location="appendix.application",
            title="Full cached MovieLens robustness suite",
            source_path_hint="application/outputs/tors_robustness_full_v1/robustness_summary.csv",
            note="Full-replication cached candidate-set-size, reward-threshold, and logging-floor robustness variants.",
        ),
    ]
    return [spec.to_dict() for spec in specs]


def output_contract_rows(config: ApplicationConfig) -> list[dict]:
    return [
        {
            "artifact_id": "dataset_manifest",
            "relative_path": "dataset_manifest.csv",
            "description": "Dataset counts, file contract, and preparation status.",
        },
        {
            "artifact_id": "dataset_provenance",
            "relative_path": "dataset_provenance.csv",
            "description": "Official source URL, archive checksum when present, and extraction file manifest.",
        },
        {
            "artifact_id": "generator_manifest",
            "relative_path": "generator_manifest.csv",
            "description": "Locked stage-1 generator library and support interpretation.",
        },
        {
            "artifact_id": "comparator_manifest",
            "relative_path": "comparator_manifest.csv",
            "description": "Shared application comparator family aligned with simulation.",
        },
        {
            "artifact_id": "summary",
            "relative_path": "summary.csv",
            "description": "Main application comparator summary once runs begin.",
        },
        {
            "artifact_id": "summary_full",
            "relative_path": "summary_full.csv",
            "description": "Full application comparator and ablation rows across replications.",
        },
        {
            "artifact_id": "selection_frequency",
            "relative_path": "selection_frequency.csv",
            "description": "Selected-policy mode frequencies across replicated split runs.",
        },
        {
            "artifact_id": "policy_delta_generators",
            "relative_path": "policy_delta_generators.csv",
            "description": "Generator-level CASP versus DR-value-only differences.",
        },
        {
            "artifact_id": "policy_delta_items",
            "relative_path": "policy_delta_items.csv",
            "description": "Item-level differences inside the induced feasible set.",
        },
        {
            "artifact_id": "support_frontier",
            "relative_path": "diagnostics/value_burden_frontier.csv",
            "description": "Support-aware application frontier diagnostic.",
        },
        {
            "artifact_id": "lambda_sensitivity",
            "relative_path": "../../figures/application_assets/lambda_sensitivity.csv",
            "description": "CASP lambda-sensitivity asset generated from the application summary.",
        },
        {
            "artifact_id": "weight_diagnostics",
            "relative_path": "../../figures/application_assets/appendix_application_weight_diagnostics_table.tex",
            "description": "ESS and max-importance-weight diagnostics for the application comparators.",
        },
        {
            "artifact_id": "robustness_full_cached",
            "relative_path": "tors_robustness_full_v1/robustness_summary.csv",
            "description": "Full cached MovieLens robustness-suite summary.",
        },
        {
            "artifact_id": "logging_diagnostics",
            "relative_path": "logging_diagnostics.csv",
            "description": "Appendix-facing support and logging concentration diagnostics.",
        },
        {
            "artifact_id": "run_manifest",
            "relative_path": "run_manifest.csv",
            "description": "Execution mode contract with holdout, replication count, and evaluation context budget.",
        },
        {
            "artifact_id": "figure_table_map",
            "relative_path": "figure_table_map.csv",
            "description": "Machine-readable manuscript asset map.",
        },
    ]


def comparator_manifest_rows() -> list[dict]:
    return [spec.to_dict() for spec in application_comparators()]
