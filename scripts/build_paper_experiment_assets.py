from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List


ROOT = Path(__file__).resolve().parents[1]
SUITE_ROOT = ROOT / "outputs" / "runs" / "simulation" / "phase3_external_baselines_full"
OUT_DIR = ROOT / "outputs" / "paper_assets" / "simulation" / "phase3_external_baselines"
PRECISION_SUITE_ROOT = ROOT / "outputs" / "runs" / "simulation" / "phase3_block5_precision_followup"
PRECISION_OUT_DIR = ROOT / "outputs" / "paper_assets" / "simulation" / "phase3_block5_precision_followup"


def mean(values: Iterable[float]) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0


def load_summary(block: str) -> List[Dict[str, str]]:
    path = SUITE_ROOT / block / "summary.csv"
    with path.open() as handle:
        return list(csv.DictReader(handle))


def load_precision_block5_summary() -> List[Dict[str, str]]:
    path = PRECISION_SUITE_ROOT / "block5_sample_size" / "summary.csv"
    with path.open() as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fieldnames})


def format_label(comparator: str) -> str:
    mapping = {
        "stagewise_proxy": "Stagewise",
        "plugin_reward": "Plug-in",
        "dr_value_only": "DR value only",
        "dr_lcb_beta_0.50": "DR-LCB 0.50",
        "casp_lambda_0.050": "CASP 0.05",
        "ma_style_two_stage_opl": "Ma-style OPL",
        "wang_style_downstream_generator": "Wang-style generator",
        "oracle": "Oracle",
    }
    return mapping[comparator]


def build_block1_table(rows: List[Dict[str, str]]) -> str:
    keep = [
        "stagewise_proxy",
        "dr_value_only",
        "casp_lambda_0.050",
        "ma_style_two_stage_opl",
        "wang_style_downstream_generator",
        "oracle",
    ]
    filtered = {row["comparator"]: row for row in rows if row["comparator"] in keep}
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{@{}lrrr@{}}",
        r"\toprule",
        r"Method & Value & Regret & Burden \\",
        r"\midrule",
    ]
    for comparator in keep:
        row = filtered[comparator]
        lines.append(
            f"{format_label(comparator)} & "
            f"{float(row['true_value_mean']):.4f} & "
            f"{float(row['oracle_regret_mean']):.4f} & "
            f"{float(row['support_burden_mean']):.2f} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Block 1 counterexample. The stagewise proxy selects the wrong generator, while every end-to-end learner recovers the optimum.}",
            r"\label{tab:block1-counterexample}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines) + "\n"


def build_crossblock_table() -> str:
    blocks = [
        "block2_coupling",
        "block3_support",
        "block4_large_action",
        "block5_sample_size",
    ]
    keep = [
        "stagewise_proxy",
        "plugin_reward",
        "dr_value_only",
        "dr_lcb_beta_0.50",
        "casp_lambda_0.050",
        "ma_style_two_stage_opl",
        "wang_style_downstream_generator",
    ]
    stats: Dict[str, Dict[str, float]] = {
        comparator: {"value": 0.0, "regret": 0.0, "burden": 0.0, "count": 0.0}
        for comparator in keep
    }
    for block in blocks:
        rows = load_summary(block)
        for comparator in keep:
            subset = [row for row in rows if row["comparator"] == comparator]
            stats[comparator]["value"] += mean(float(row["true_value_mean"]) for row in subset)
            stats[comparator]["regret"] += mean(float(row["oracle_regret_mean"]) for row in subset)
            stats[comparator]["burden"] += mean(float(row["support_burden_mean"]) for row in subset)
            stats[comparator]["count"] += 1.0

    best_value = max(stats[comp]["value"] / stats[comp]["count"] for comp in keep)
    best_regret = min(stats[comp]["regret"] / stats[comp]["count"] for comp in keep)
    best_burden = min(stats[comp]["burden"] / stats[comp]["count"] for comp in keep)

    def maybe_bold(value: float, target: float, decimals: int) -> str:
        formatted = f"{value:.{decimals}f}"
        if abs(value - target) < 1e-9:
            return rf"\textbf{{{formatted}}}"
        return formatted

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{@{}lrrr@{}}",
        r"\toprule",
        r"Method & Avg.\ value & Avg.\ regret & Avg.\ burden \\",
        r"\midrule",
    ]
    for comparator in keep:
        count = stats[comparator]["count"]
        value = stats[comparator]["value"] / count
        regret = stats[comparator]["regret"] / count
        burden = stats[comparator]["burden"] / count
        lines.append(
            f"{format_label(comparator)} & "
            f"{maybe_bold(value, best_value, 4)} & "
            f"{maybe_bold(regret, best_regret, 4)} & "
            f"{maybe_bold(burden, best_burden, 2)} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Cross-block averages over Blocks 2--5. Ma-style OPL is the strongest raw-value external comparator on average, while \CASP{} is the lowest-burden learned selector.}",
            r"\label{tab:crossblock-summary}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines) + "\n"


def build_block_comparator_table(block: str, rows: List[Dict[str, str]]) -> str:
    keep = [
        "stagewise_proxy",
        "plugin_reward",
        "dr_value_only",
        "dr_lcb_beta_0.50",
        "casp_lambda_0.050",
        "ma_style_two_stage_opl",
        "wang_style_downstream_generator",
        "oracle",
    ]
    title_map = {
        "block1_counterexample": "Block 1 full comparator table",
        "block2_coupling": "Block 2 full comparator table",
        "block3_support": "Block 3 full comparator table",
        "block4_large_action": "Block 4 full comparator table",
        "block5_sample_size": "Block 5 full comparator table",
    }
    label_map = {
        "block1_counterexample": "tab:app-block1-full",
        "block2_coupling": "tab:app-block2-full",
        "block3_support": "tab:app-block3-full",
        "block4_large_action": "tab:app-block4-full",
        "block5_sample_size": "tab:app-block5-full",
    }
    filtered = {row["comparator"]: row for row in rows if row["comparator"] in keep}
    direct_block = block == "block1_counterexample"
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{@{}lrrrr@{}}",
        r"\toprule",
        r"Method & Value & Regret & Burden & Stability \\",
        r"\midrule",
    ]
    for comparator in keep:
        if direct_block:
            row = filtered[comparator]
            value = float(row["true_value_mean"])
            regret = float(row["oracle_regret_mean"])
            burden = float(row["support_burden_mean"])
            stability = float(row["selected_policy_mode_frequency"] or 0.0)
        else:
            subset = [row for row in rows if row["comparator"] == comparator]
            value = mean(float(row["true_value_mean"]) for row in subset)
            regret = mean(float(row["oracle_regret_mean"]) for row in subset)
            burden = mean(float(row["support_burden_mean"]) for row in subset)
            stability = mean(float(row["selected_policy_mode_frequency"] or 0.0) for row in subset)
        lines.append(
            f"{format_label(comparator)} & "
            f"{value:.4f} & "
            f"{regret:.4f} & "
            f"{burden:.2f} & "
            f"{stability:.3f} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            f"\\caption{{{title_map[block]}. For Blocks 2--5 the entries are sweep averages; Block 1 is reported directly.}}",
            f"\\label{{{label_map[block]}}}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines) + "\n"


def build_block2_coupling_csv(rows: List[Dict[str, str]]) -> List[Dict[str, object]]:
    keep = [
        "stagewise_proxy",
        "dr_value_only",
        "dr_lcb_beta_0.50",
        "casp_lambda_0.050",
        "ma_style_two_stage_opl",
    ]
    grouped: Dict[float, Dict[str, Dict[str, float]]] = {}
    for row in rows:
        comparator = row["comparator"]
        if comparator not in keep:
            continue
        sweep_value = float(row["sweep_value"])
        grouped.setdefault(sweep_value, {})[comparator] = {
            "value": float(row["true_value_mean"]),
            "burden": float(row["support_burden_mean"]),
        }

    out_rows: List[Dict[str, object]] = []
    for sweep_value in sorted(grouped):
        metrics = grouped[sweep_value]
        out_rows.append(
            {
                "sweep_value": sweep_value,
                "stagewise_value": metrics["stagewise_proxy"]["value"],
                "dr_value_only_value": metrics["dr_value_only"]["value"],
                "dr_lcb_value": metrics["dr_lcb_beta_0.50"]["value"],
                "casp_value": metrics["casp_lambda_0.050"]["value"],
                "ma_style_value": metrics["ma_style_two_stage_opl"]["value"],
                "stagewise_burden": metrics["stagewise_proxy"]["burden"],
                "dr_value_only_burden": metrics["dr_value_only"]["burden"],
                "dr_lcb_burden": metrics["dr_lcb_beta_0.50"]["burden"],
                "casp_burden": metrics["casp_lambda_0.050"]["burden"],
                "ma_style_burden": metrics["ma_style_two_stage_opl"]["burden"],
            }
        )
    return out_rows


def build_ablation_summary_table() -> str:
    blocks = [
        ("block2_coupling", "B2"),
        ("block3_support", "B3"),
        ("block4_large_action", "B4"),
        ("block5_sample_size", "B5"),
    ]
    keep = [
        ("casp_ablation_normalized_full", "Normalized full"),
        ("casp_ablation_normalized_stage1_only", "Normalized stage-1 only"),
        ("casp_ablation_normalized_stage2_only", "Normalized stage-2 only"),
        ("casp_ablation_raw_full", "Raw full"),
    ]
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{@{}llrrrr@{}}",
        r"\toprule",
        r"Block & Ablation & Avg.\ value & Avg.\ regret & Avg.\ burden & Avg.\ stability \\",
        r"\midrule",
    ]
    for block_name, block_label in blocks:
        rows = load_summary(block_name)
        for comparator, label in keep:
            subset = [row for row in rows if row["comparator"] == comparator]
            lines.append(
                f"{block_label} & {label} & "
                f"{mean(float(row['true_value_mean']) for row in subset):.4f} & "
                f"{mean(float(row['oracle_regret_mean']) for row in subset):.4f} & "
                f"{mean(float(row['support_burden_mean']) for row in subset):.2f} & "
                f"{mean(float(row['selected_policy_mode_frequency'] or 0.0) for row in subset):.3f} \\\\"
            )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{CASP ablation summary across Blocks 2--5. The normalized full burden is the main ablation baseline; the stage-1-only and stage-2-only variants isolate which component of the support burden carries the empirical signal; the raw full burden documents the pre-calibration failure mode.}",
            r"\label{tab:app-ablation-summary}",
            r"\end{table*}",
        ]
    )
    return "\n".join(lines) + "\n"


def build_frontier_csv() -> List[Dict[str, object]]:
    blocks = [
        ("block2_coupling", "B2"),
        ("block3_support", "B3"),
        ("block4_large_action", "B4"),
        ("block5_sample_size", "B5"),
    ]
    keep = [
        "casp_lambda_0.050",
        "dr_value_only",
        "dr_lcb_beta_0.50",
        "ma_style_two_stage_opl",
    ]
    out_rows: List[Dict[str, object]] = []
    for block, label in blocks:
        rows = load_summary(block)
        averages: Dict[str, Dict[str, float]] = {}
        for comparator in keep:
            subset = [row for row in rows if row["comparator"] == comparator]
            averages[comparator] = {
                "value": mean(float(row["true_value_mean"]) for row in subset),
                "burden": mean(float(row["support_burden_mean"]) for row in subset),
                "stability": mean(float(row["selected_policy_mode_frequency"] or 0.0) for row in subset),
            }
        out_rows.append(
            {
                "block_label": label,
                "casp_value": averages["casp_lambda_0.050"]["value"],
                "casp_burden": averages["casp_lambda_0.050"]["burden"],
                "casp_stability": averages["casp_lambda_0.050"]["stability"],
                "dr_value_only_value": averages["dr_value_only"]["value"],
                "dr_value_only_burden": averages["dr_value_only"]["burden"],
                "dr_value_only_stability": averages["dr_value_only"]["stability"],
                "dr_lcb_value": averages["dr_lcb_beta_0.50"]["value"],
                "dr_lcb_burden": averages["dr_lcb_beta_0.50"]["burden"],
                "dr_lcb_stability": averages["dr_lcb_beta_0.50"]["stability"],
                "ma_style_value": averages["ma_style_two_stage_opl"]["value"],
                "ma_style_burden": averages["ma_style_two_stage_opl"]["burden"],
                "ma_style_stability": averages["ma_style_two_stage_opl"]["stability"],
            }
        )
    return out_rows


def build_ablation_frontier_csv() -> List[Dict[str, object]]:
    blocks = [
        ("block2_coupling", "B2"),
        ("block3_support", "B3"),
        ("block4_large_action", "B4"),
        ("block5_sample_size", "B5"),
    ]
    keep = [
        "casp_ablation_normalized_full",
        "casp_ablation_normalized_stage1_only",
        "casp_ablation_normalized_stage2_only",
        "casp_ablation_raw_full",
    ]
    rows_out: List[Dict[str, object]] = []
    for block_name, block_label in blocks:
        rows = load_summary(block_name)
        averages: Dict[str, Dict[str, float]] = {}
        for comparator in keep:
            subset = [row for row in rows if row["comparator"] == comparator]
            averages[comparator] = {
                "value": mean(float(row["true_value_mean"]) for row in subset),
                "burden": mean(float(row["support_burden_mean"]) for row in subset),
                "stability": mean(float(row["selected_policy_mode_frequency"] or 0.0) for row in subset),
            }
        rows_out.append(
            {
                "block_label": block_label,
                "normalized_full_value": averages["casp_ablation_normalized_full"]["value"],
                "normalized_full_burden": averages["casp_ablation_normalized_full"]["burden"],
                "normalized_full_stability": averages["casp_ablation_normalized_full"]["stability"],
                "stage1_only_value": averages["casp_ablation_normalized_stage1_only"]["value"],
                "stage1_only_burden": averages["casp_ablation_normalized_stage1_only"]["burden"],
                "stage1_only_stability": averages["casp_ablation_normalized_stage1_only"]["stability"],
                "stage2_only_value": averages["casp_ablation_normalized_stage2_only"]["value"],
                "stage2_only_burden": averages["casp_ablation_normalized_stage2_only"]["burden"],
                "stage2_only_stability": averages["casp_ablation_normalized_stage2_only"]["stability"],
                "raw_full_value": averages["casp_ablation_raw_full"]["value"],
                "raw_full_burden": averages["casp_ablation_raw_full"]["burden"],
                "raw_full_stability": averages["casp_ablation_raw_full"]["stability"],
            }
        )
    return rows_out


def precision_suite_runtime_text() -> str:
    suite_info_path = PRECISION_SUITE_ROOT / "suite_info.json"
    payload = json.loads(suite_info_path.read_text(encoding="utf-8"))
    created = payload["created_at"]
    finished = payload["finished_at"]
    return f"{created} to {finished}"


def build_block5_precision_table(rows: List[Dict[str, str]]) -> str:
    keep = [
        "stagewise_proxy",
        "plugin_reward",
        "dr_value_only",
        "dr_lcb_beta_0.50",
        "casp_lambda_0.050",
        "ma_style_two_stage_opl",
        "wang_style_downstream_generator",
        "oracle",
    ]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{@{}lrrrr@{}}",
        r"\toprule",
        r"Method & Avg.\ value & Avg.\ regret & Avg.\ burden & Avg.\ stability \\",
        r"\midrule",
    ]
    for comparator in keep:
        subset = [row for row in rows if row["comparator"] == comparator]
        lines.append(
            f"{format_label(comparator)} & "
            f"{mean(float(row['true_value_mean']) for row in subset):.4f} & "
            f"{mean(float(row['oracle_regret_mean']) for row in subset):.4f} & "
            f"{mean(float(row['support_burden_mean']) for row in subset):.2f} & "
            f"{mean(float(row['selected_policy_mode_frequency'] or 0.0) for row in subset):.3f} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Verification-focused Block 5 precision rerun over sample sizes $n\in\{600,1200,2400,4800\}$. The rerun preserves the same qualitative story as the locked external-baseline suite: Ma-style OPL remains strongest on raw value, while \CASP{} remains the lowest-burden learned selector.}",
            r"\label{tab:block5-precision-verification}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines) + "\n"


def build_block5_precision_csv(rows: List[Dict[str, str]]) -> List[Dict[str, object]]:
    keep = [
        "stagewise_proxy",
        "plugin_reward",
        "dr_value_only",
        "dr_lcb_beta_0.50",
        "casp_lambda_0.050",
        "ma_style_two_stage_opl",
        "wang_style_downstream_generator",
        "oracle",
    ]
    out_rows: List[Dict[str, object]] = []
    for comparator in keep:
        subset = [row for row in rows if row["comparator"] == comparator]
        out_rows.append(
            {
                "comparator": comparator,
                "label": format_label(comparator),
                "avg_value": mean(float(row["true_value_mean"]) for row in subset),
                "avg_regret": mean(float(row["oracle_regret_mean"]) for row in subset),
                "avg_burden": mean(float(row["support_burden_mean"]) for row in subset),
                "avg_stability": mean(float(row["selected_policy_mode_frequency"] or 0.0) for row in subset),
            }
        )
    return out_rows


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PRECISION_OUT_DIR.mkdir(parents=True, exist_ok=True)

    block1_rows = load_summary("block1_counterexample")
    block2_rows = load_summary("block2_coupling")
    block3_rows = load_summary("block3_support")
    block4_rows = load_summary("block4_large_action")
    block5_rows = load_summary("block5_sample_size")
    precision_block5_rows = load_precision_block5_summary()

    (OUT_DIR / "block1_counterexample_table.tex").write_text(build_block1_table(block1_rows))
    (OUT_DIR / "crossblock_summary_table.tex").write_text(build_crossblock_table())
    (OUT_DIR / "appendix_block1_full_table.tex").write_text(build_block_comparator_table("block1_counterexample", block1_rows))
    (OUT_DIR / "appendix_block2_full_table.tex").write_text(build_block_comparator_table("block2_coupling", block2_rows))
    (OUT_DIR / "appendix_block3_full_table.tex").write_text(build_block_comparator_table("block3_support", block3_rows))
    (OUT_DIR / "appendix_block4_full_table.tex").write_text(build_block_comparator_table("block4_large_action", block4_rows))
    (OUT_DIR / "appendix_block5_full_table.tex").write_text(build_block_comparator_table("block5_sample_size", block5_rows))
    (OUT_DIR / "appendix_ablation_summary_table.tex").write_text(build_ablation_summary_table())

    write_csv(
        OUT_DIR / "block2_coupling_key.csv",
        build_block2_coupling_csv(block2_rows),
        [
            "sweep_value",
            "stagewise_value",
            "dr_value_only_value",
            "dr_lcb_value",
            "casp_value",
            "ma_style_value",
            "stagewise_burden",
            "dr_value_only_burden",
            "dr_lcb_burden",
            "casp_burden",
            "ma_style_burden",
        ],
    )
    write_csv(
        OUT_DIR / "frontier_key.csv",
        build_frontier_csv(),
        [
            "block_label",
            "casp_value",
            "casp_burden",
            "casp_stability",
            "dr_value_only_value",
            "dr_value_only_burden",
            "dr_value_only_stability",
            "dr_lcb_value",
            "dr_lcb_burden",
            "dr_lcb_stability",
            "ma_style_value",
            "ma_style_burden",
            "ma_style_stability",
        ],
    )
    write_csv(
        OUT_DIR / "ablation_frontier_key.csv",
        build_ablation_frontier_csv(),
        [
            "block_label",
            "normalized_full_value",
            "normalized_full_burden",
            "normalized_full_stability",
            "stage1_only_value",
            "stage1_only_burden",
            "stage1_only_stability",
            "stage2_only_value",
            "stage2_only_burden",
            "stage2_only_stability",
            "raw_full_value",
            "raw_full_burden",
            "raw_full_stability",
        ],
    )

    manifest_rows = [
        {
            "artifact": "block1_counterexample_table.tex",
            "description": "Generated Block 1 table for the main paper.",
        },
        {
            "artifact": "crossblock_summary_table.tex",
            "description": "Generated cross-block summary table for the main paper.",
        },
        {
            "artifact": "block2_coupling_key.csv",
            "description": "Wide-format Block 2 coupling data for the headline figure.",
        },
        {
            "artifact": "frontier_key.csv",
            "description": "Wide-format frontier data for value-burden and stability-burden figures.",
        },
        {
            "artifact": "appendix_block1_full_table.tex",
            "description": "Appendix table with the full Block 1 comparator set.",
        },
        {
            "artifact": "appendix_block2_full_table.tex",
            "description": "Appendix table with the full Block 2 comparator set.",
        },
        {
            "artifact": "appendix_block3_full_table.tex",
            "description": "Appendix table with the full Block 3 comparator set.",
        },
        {
            "artifact": "appendix_block4_full_table.tex",
            "description": "Appendix table with the full Block 4 comparator set.",
        },
        {
            "artifact": "appendix_block5_full_table.tex",
            "description": "Appendix table with the full Block 5 comparator set.",
        },
        {
            "artifact": "appendix_ablation_summary_table.tex",
            "description": "Appendix table summarizing CASP ablations across Blocks 2-5.",
        },
        {
            "artifact": "ablation_frontier_key.csv",
            "description": "Wide-format appendix data for ablation frontier diagnostics.",
        },
    ]
    write_csv(
        OUT_DIR / "manifest.csv",
        manifest_rows,
        ["artifact", "description"],
    )

    (PRECISION_OUT_DIR / "block5_precision_table.tex").write_text(
        build_block5_precision_table(precision_block5_rows),
        encoding="utf-8",
    )
    write_csv(
        PRECISION_OUT_DIR / "block5_precision_summary.csv",
        build_block5_precision_csv(precision_block5_rows),
        ["comparator", "label", "avg_value", "avg_regret", "avg_burden", "avg_stability"],
    )
    write_csv(
        PRECISION_OUT_DIR / "manifest.csv",
        [
            {
                "artifact": "block5_precision_table.tex",
                "description": "Compact TeX table for the verification-focused Block 5 precision rerun.",
            },
            {
                "artifact": "block5_precision_summary.csv",
                "description": f"CSV summary for the Block 5 precision rerun ({precision_suite_runtime_text()}).",
            },
        ],
        ["artifact", "description"],
    )


if __name__ == "__main__":
    main()
