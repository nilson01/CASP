from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUITE_ROOT = (
    ROOT
    / "outputs"
    / "runs"
    / "application"
    / "tors_full_movielens_1m_rebalanced_v2_eps0p10_tau1p00"
)
DEFAULT_ROBUSTNESS_ROOT = ROOT / "outputs" / "runs" / "application" / "tors_robustness_full_v1"
OUT_DIR = ROOT / "outputs" / "paper_assets" / "application"
GENERATOR_LABELS = {
    0: "Popularity",
    1: "Genre",
    2: "Collaborative",
    3: "LongTail",
}
COMPARATOR_LABELS = {
    "behavior": "Behavior",
    "random_uniform": "Random uniform",
    "reconstructed_oracle": "Reconstructed oracle",
    "stagewise_proxy": "Stagewise proxy",
    "plugin_reward": "Plug-in reward",
    "dr_value_only": "DR value only",
    "dr_lcb_beta_0.50": r"DR-LCB ($\beta=0.50$)",
    "casp_lambda_0.000": r"\CASP{} ($\lambda=0$)",
    "casp_lambda_0.010": r"\CASP{} ($\lambda=0.01$)",
    "casp_lambda_0.020": r"\CASP{} ($\lambda=0.02$)",
    "casp_lambda_0.050": r"\CASP{} ($\lambda=0.05$)",
    "casp_lambda_0.100": r"\CASP{} ($\lambda=0.10$)",
    "casp_lambda_0.200": r"\CASP{} ($\lambda=0.20$)",
    "ma_style_two_stage_opl": "Ma-style OPL",
    "wang_style_downstream_generator": "Wang-style generator",
    "casp_ablation_normalized_full": "Normalized full",
    "casp_ablation_normalized_stage1_only": "Normalized stage-1 only",
    "casp_ablation_normalized_stage2_only": "Normalized stage-2 only",
    "casp_ablation_raw_full": "Raw full",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def f(row: dict[str, str], field: str, default: float = 0.0) -> float:
    value = row.get(field, "")
    return float(value) if value not in ("", None) else default


def mean(values: Iterable[float]) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0


def fmt_fixed(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def fmt_metric(value: float, digits: int = 3) -> str:
    if abs(value) >= 100000:
        exponent = 0
        mantissa = float(value)
        while abs(mantissa) >= 10:
            mantissa /= 10
            exponent += 1
        return rf"${mantissa:.2f}\times 10^{exponent}$"
    if abs(value) >= 1000:
        return f"{value:.1f}"
    if abs(value) >= 100:
        return f"{value:.1f}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    return f"{value:.{digits}f}"


def tex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
    }
    for raw, escaped in replacements.items():
        text = text.replace(raw, escaped)
    return text


def by_comparator(summary_rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["comparator"]: row for row in summary_rows}


def mode_freq(row: dict[str, str]) -> str:
    value = f(row, "selected_policy_mode_frequency")
    return "--" if value <= 0 else f"{value:.2f}"


def build_main_comparator_table(summary_rows: list[dict[str, str]]) -> str:
    rows = by_comparator(summary_rows)
    keep = [
        "casp_lambda_0.050",
        "dr_value_only",
        "dr_lcb_beta_0.50",
        "ma_style_two_stage_opl",
        "wang_style_downstream_generator",
    ]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Comparator & DR value & Burden & ESS & Max $w$ \\",
        r"\midrule",
    ]
    for comparator in keep:
        row = rows[comparator]
        lines.append(
            f"{COMPARATOR_LABELS[comparator]} & "
            f"{fmt_fixed(f(row, 'dr_value_mean'), 3)} & "
            f"{fmt_metric(f(row, 'support_burden_mean'), 1)} & "
            f"{fmt_metric(f(row, 'ess_proxy_mean'), 1)} & "
            f"{fmt_metric(f(row, 'max_importance_weight_mean'), 1)} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Main comparator summary for the reconstructed \texttt{MovieLens 1M} application. Higher DR value and ESS are better; lower burden and maximum importance weight are better.}",
            r"\label{tab:mlappmain}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines) + "\n"


def build_full_comparator_table(summary_rows: list[dict[str, str]]) -> str:
    rows = by_comparator(summary_rows)
    keep = [
        "behavior",
        "random_uniform",
        "reconstructed_oracle",
        "stagewise_proxy",
        "plugin_reward",
        "dr_value_only",
        "dr_lcb_beta_0.50",
        "casp_lambda_0.050",
        "ma_style_two_stage_opl",
        "wang_style_downstream_generator",
    ]
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Comparator & Oracle & DR & Burden & ESS & Max $w$ & Mode freq. \\",
        r"\midrule",
    ]
    for comparator in keep:
        row = rows[comparator]
        lines.append(
            f"{COMPARATOR_LABELS[comparator]} & "
            f"{fmt_fixed(f(row, 'oracle_value_mean'), 3)} & "
            f"{fmt_fixed(f(row, 'dr_value_mean'), 3)} & "
            f"{fmt_metric(f(row, 'support_burden_mean'), 1)} & "
            f"{fmt_metric(f(row, 'ess_proxy_mean'), 1)} & "
            f"{fmt_metric(f(row, 'max_importance_weight_mean'), 1)} & "
            f"{mode_freq(row)} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Full application comparator summary for the accepted reconstructed \texttt{MovieLens 1M} run, including effective-sample-size and tail-weight diagnostics.}",
            r"\label{tab:app-full-comparators}",
            r"\end{table*}",
        ]
    )
    return "\n".join(lines) + "\n"


def build_ablation_table(summary_rows: list[dict[str, str]]) -> str:
    rows = by_comparator(summary_rows)
    keep = [
        "casp_ablation_normalized_full",
        "casp_ablation_normalized_stage1_only",
        "casp_ablation_normalized_stage2_only",
        "casp_ablation_raw_full",
    ]
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Ablation & Oracle & DR & Burden & ESS & Max $w$ & Unique policies \\",
        r"\midrule",
    ]
    for comparator in keep:
        row = rows[comparator]
        lines.append(
            f"{COMPARATOR_LABELS[comparator]} & "
            f"{fmt_fixed(f(row, 'oracle_value_mean'), 3)} & "
            f"{fmt_fixed(f(row, 'dr_value_mean'), 3)} & "
            f"{fmt_metric(f(row, 'support_burden_mean'), 1)} & "
            f"{fmt_metric(f(row, 'ess_proxy_mean'), 1)} & "
            f"{fmt_metric(f(row, 'max_importance_weight_mean'), 1)} & "
            f"{int(f(row, 'selected_policy_unique_count'))} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Application ablation summary at $\lambda=0.05$. The raw-full variant is retained as an appendix calibration sensitivity rather than a headline main-text result.}",
            r"\label{tab:app-ablation}",
            r"\end{table*}",
        ]
    )
    return "\n".join(lines) + "\n"


def lambda_value(comparator: str) -> float:
    return float(comparator.removeprefix("casp_lambda_"))


def lambda_sensitivity_rows(summary_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    rows = []
    for row in summary_rows:
        comparator = row["comparator"]
        if not comparator.startswith("casp_lambda_"):
            continue
        rows.append(
            {
                "lambda": lambda_value(comparator),
                "label": comparator.replace("casp_lambda_", "lambda="),
                "dr_value": f(row, "dr_value_mean"),
                "support_burden": f(row, "support_burden_mean"),
                "ess_proxy": f(row, "ess_proxy_mean"),
                "max_importance_weight": f(row, "max_importance_weight_mean"),
                "mode_frequency": f(row, "selected_policy_mode_frequency"),
                "unique_policies": int(f(row, "selected_policy_unique_count")),
            }
        )
    return sorted(rows, key=lambda row: row["lambda"])


def build_lambda_table(rows: list[dict[str, object]]) -> str:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{rrrrrr}",
        r"\toprule",
        r"$\lambda$ & DR value & Burden & ESS & Max $w$ & Mode freq. \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{float(row['lambda']):.3f} & "
            f"{fmt_fixed(float(row['dr_value']), 3)} & "
            f"{fmt_metric(float(row['support_burden']), 1)} & "
            f"{fmt_metric(float(row['ess_proxy']), 1)} & "
            f"{fmt_metric(float(row['max_importance_weight']), 1)} & "
            f"{float(row['mode_frequency']):.2f} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{\CASP{} $\lambda$-sensitivity on the reconstructed \texttt{MovieLens 1M} application. The grid is generated from the same completed application run as the main comparator table.}",
            r"\label{tab:app-lambda-sensitivity}",
            r"\end{table*}",
        ]
    )
    return "\n".join(lines) + "\n"


def build_weight_diagnostics_table(summary_rows: list[dict[str, str]]) -> str:
    rows = by_comparator(summary_rows)
    keep = [
        "casp_lambda_0.050",
        "dr_value_only",
        "dr_lcb_beta_0.50",
        "ma_style_two_stage_opl",
        "wang_style_downstream_generator",
    ]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Comparator & Burden & ESS & Max $w$ \\",
        r"\midrule",
    ]
    for comparator in keep:
        row = rows[comparator]
        lines.append(
            f"{COMPARATOR_LABELS[comparator]} & "
            f"{fmt_metric(f(row, 'support_burden_mean'), 1)} & "
            f"{fmt_metric(f(row, 'ess_proxy_mean'), 1)} & "
            f"{fmt_metric(f(row, 'max_importance_weight_mean'), 1)} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Application effective-sample-size and tail-weight diagnostics. These diagnostics separate the support-burden signal from conventional ESS and maximum-weight summaries, which are reported for calibration rather than used as the headline claim.}",
            r"\label{tab:app-weight-diagnostics}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines) + "\n"


def processed_root_for_suite(suite_root: Path) -> Path:
    config = read_json(suite_root / "application_config.json")
    return ROOT / "data" / "processed" / config["name"]


def stage1_zero_support_profile(suite_root: Path) -> list[dict[str, object]]:
    config = read_json(suite_root / "application_config.json")
    processed_root = processed_root_for_suite(suite_root)
    context_limit = int(config.get("context_limit", 25000))
    holdout_fraction = float(config.get("holdout_fraction", 0.2))
    contexts = []
    with (processed_root / "request_contexts.csv").open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            contexts.append(row)
            if len(contexts) >= context_limit:
                break
    eval_size = max(1, int(round(len(contexts) * holdout_fraction)))
    eval_rows = contexts[-eval_size:]
    zero_counts = Counter()
    for row in eval_rows:
        for generator, prob in enumerate(json.loads(row["stage1_probs_json"])):
            if float(prob) <= 0.0:
                zero_counts[generator] += 1

    policy_shares = {
        row["generator"]: row
        for row in build_policy_delta_generator_shares(suite_root)
    }
    rows = []
    for generator in range(4):
        label = GENERATOR_LABELS[generator]
        share_row = policy_shares[label]
        rows.append(
            {
                "generator": label,
                "zero_stage1_support_share": zero_counts[generator] / max(len(eval_rows), 1),
                "dr_selected_share": share_row["dr_share"],
                "casp_selected_share": share_row["casp_share"],
            }
        )
    return rows


def offsupport_rows(summary_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    rows = by_comparator(summary_rows)
    keep = [
        "casp_lambda_0.050",
        "dr_value_only",
        "dr_lcb_beta_0.50",
        "ma_style_two_stage_opl",
        "wang_style_downstream_generator",
        "stagewise_proxy",
        "casp_ablation_raw_full",
    ]
    out = []
    for comparator in keep:
        burden = f(rows[comparator], "support_burden_mean")
        out.append(
            {
                "comparator": COMPARATOR_LABELS[comparator],
                "support_burden": burden,
                "floor_implied_offsupport_mass": min(max(burden / 1e9, 0.0), 1.0),
            }
        )
    return out


def fmt_share(value: float) -> str:
    if value < 0.001:
        return r"$<0.001$"
    return f"{value:.3f}"


def build_support_violation_table(
    summary_rows: list[dict[str, str]],
    zero_support_rows: list[dict[str, object]],
) -> str:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Selector & Burden & Floor-implied off-support mass \\",
        r"\midrule",
    ]
    for row in offsupport_rows(summary_rows):
        lines.append(
            f"{row['comparator']} & "
            f"{fmt_metric(float(row['support_burden']), 1)} & "
            f"{fmt_share(float(row['floor_implied_offsupport_mass']))} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\vspace{0.7em}",
            r"\begin{tabular}{lrrr}",
            r"\toprule",
            r"Generator & Zero stage-1 support share & DR-only selected share & \CASP{} selected share \\",
            r"\midrule",
        ]
    )
    for row in zero_support_rows:
        lines.append(
            f"{row['generator']} & "
            f"{float(row['zero_stage1_support_share']):.3f} & "
            f"{float(row['dr_selected_share']):.3f} & "
            f"{float(row['casp_selected_share']):.3f} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Application support-violation diagnostics. The first panel reports the burden-scale implication of the $10^{-9}$ denominator floor: values near $0.65$ mean the selector frequently asks for action pairs outside reconstructed support. The second panel explains the main source of that behavior: DR-value-only concentrates on the collaborative generator, which has zero reconstructed stage-1 support on many evaluation contexts, while \CASP{} redistributes mass toward supported generators.}",
            r"\label{tab:app-support-violation}",
            r"\end{table*}",
        ]
    )
    return "\n".join(lines) + "\n"


def build_frontier_csv(summary_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    rows = by_comparator(summary_rows)
    keep = [
        ("CASP", "casp_lambda_0.050"),
        ("DR-only", "dr_value_only"),
        ("Ma-style", "ma_style_two_stage_opl"),
        ("Wang-style", "wang_style_downstream_generator"),
    ]
    return [
        {
            "label": label,
            "burden": f(rows[comparator], "support_burden_mean"),
            "value": f(rows[comparator], "dr_value_mean"),
            "stability": f(rows[comparator], "selected_policy_mode_frequency"),
            "ess_proxy": f(rows[comparator], "ess_proxy_mean"),
            "max_importance_weight": f(rows[comparator], "max_importance_weight_mean"),
        }
        for label, comparator in keep
    ]


def build_policy_delta_generator_shares(suite_root: Path) -> list[dict[str, object]]:
    rows = read_csv(suite_root / "policy_delta_generators.csv")
    dr_counts: Counter[int] = Counter()
    casp_counts: Counter[int] = Counter()
    total = 0
    for row in rows:
        count = int(row["count"])
        total += count
        dr_counts[int(row["dr_generator"])] += count
        casp_counts[int(row["casp_generator"])] += count
    total = total or 1
    return [
        {
            "generator": GENERATOR_LABELS[index],
            "dr_share": dr_counts[index] / total,
            "casp_share": casp_counts[index] / total,
        }
        for index in range(4)
    ]


def load_key_value_csv(path: Path) -> dict[str, str]:
    return {row["field"]: row["value"] for row in read_csv(path)}


def build_support_table(suite_root: Path) -> str:
    config = read_json(suite_root / "application_config.json")
    processed_root = processed_root_for_suite(suite_root)
    support = load_key_value_csv(processed_root / "support_filter_diagnostics.csv")
    logging_rows = load_key_value_csv(suite_root / "logging_diagnostics.csv")
    context_rows = read_csv(processed_root / "request_contexts.csv")
    stage1_counts = Counter(int(row["stage1_action"]) for row in context_rows)
    total = len(context_rows) or 1
    shares = [stage1_counts[index] / total for index in range(4)]
    strict_contexts = float(support.get("final_strict_contexts", support.get("strict_contexts", 0)))
    final_contexts = float(support.get("final_contexts", total))
    rows = [
        ("Final reported contexts", f"{int(final_contexts):,}"),
        (r"Strict $\ge 2$-support share", f"{strict_contexts / max(final_contexts, 1):.5f}"),
        ("Dominant stage-1 share", f"{max(shares):.5f}"),
        ("Minimum generator share", f"{min(shares):.5f}"),
        ("Mean support-generator count", f"{float(logging_rows.get('mean_support_generator_count', 0)):.3f}"),
        ("Share with exactly two support generators", f"{float(logging_rows.get('share_exactly_two_support_generators', 0)):.3f}"),
        ("Mean stage-1 entropy on eval slice", f"{float(logging_rows.get('mean_stage1_entropy', 0)):.4f}"),
        ("Maximum support-generator count", f"{int(float(logging_rows.get('max_support_generator_count', 0)))}"),
        ("Fallback singleton pool mode", support.get("support_pool_mode", "")),
    ]
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lc}",
        r"\toprule",
        r"Diagnostic & Value \\",
        r"\midrule",
    ]
    for label, value in rows:
        lines.append(f"{label} & {tex_escape(value)} \\\\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Application support and logging diagnostics for the accepted reconstructed \texttt{MovieLens 1M} pool.}",
            r"\label{tab:app-support}",
            r"\end{table*}",
        ]
    )
    return "\n".join(lines) + "\n"


def build_provenance_table(suite_root: Path) -> str:
    suite_info = read_json(suite_root / "suite_info.json")
    provenance = load_key_value_csv(suite_root / "dataset_provenance.csv")
    run_manifest = read_csv(suite_root / "run_manifest.csv")[0]
    rows = [
        ("Source dataset", "GroupLens MovieLens 1M"),
        ("Archive URL", r"\url{" + provenance.get("source_url", "https://files.grouplens.org/datasets/movielens/ml-1m.zip") + "}"),
        ("Archive SHA-256", provenance.get("archive_sha256", "--")[:20] + "..."),
        ("Raw archive present", "Yes" if provenance.get("archive_exists") == "True" else "No"),
        ("Application suite status", suite_info.get("status", "--")),
        ("Run window", f"{suite_info.get('created_at', '--')} to {suite_info.get('finished_at', '--')}"),
        ("Replications completed", f"{suite_info.get('progress', {}).get('completed_tasks', '--')} / {run_manifest.get('split_replications', '--')}"),
        ("Failures", str(suite_info.get("progress", {}).get("failed_tasks", "--"))),
    ]
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{ll}",
        r"\toprule",
        r"Field & Value \\",
        r"\midrule",
    ]
    for label, value in rows:
        lines.append(f"{label} & {value} \\\\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Application provenance and run-manifest summary for the accepted reported run.}",
            r"\label{tab:app-provenance}",
            r"\end{table*}",
        ]
    )
    return "\n".join(lines) + "\n"


def load_robustness_summary(robustness_root: Path) -> list[dict[str, str]]:
    path = robustness_root / "robustness_summary.csv"
    if not path.exists():
        return []
    return read_csv(path)


def build_robustness_table(rows: list[dict[str, str]]) -> str:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabularx}{\textwidth}{@{}Xrrrrr@{}}",
        r"\toprule",
        r"Variant & CASP DR & CASP burden & DR-only burden & Ma burden & CASP ESS \\",
        r"\midrule",
    ]
    if rows:
        for row in rows:
            lines.append(
                f"{row['variant_label']} & "
                f"{fmt_fixed(float(row['casp_dr_value_mean']), 3)} & "
                f"{fmt_metric(float(row['casp_support_burden_mean']), 1)} & "
                f"{fmt_metric(float(row['dr_only_support_burden_mean']), 1)} & "
                f"{fmt_metric(float(row['ma_style_support_burden_mean']), 1)} & "
                f"{fmt_metric(float(row['casp_ess_proxy_mean']), 1)} \\\\"
            )
    else:
        lines.append(r"Pending full cached robustness suite & -- & -- & -- & -- & -- \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabularx}",
            r"\caption{Full cached MovieLens robustness suite. Each row reruns the full comparator layer on a cached or derived local sensitivity variant of the accepted reconstructed pool.}",
            r"\label{tab:app-robustness-full}",
            r"\end{table*}",
        ]
    )
    return "\n".join(lines) + "\n"


def build_reproducibility_audit_table(suite_root: Path, robustness_root: Path) -> str:
    rows = [
        (
            r"Table~\ref{tab:block1-counterexample}",
            r"\path{figures/phase3_external_baselines/block1_counterexample_table.tex}",
            r"\path{simulation/outputs/phase3_external_baselines_full/block1_counterexample/summary.csv}",
            r"\path{scripts/build_paper_experiment_assets.py}; run root \path{simulation/outputs/phase3_external_baselines_full}",
        ),
        (
            r"Table~\ref{tab:crossblock-summary}",
            r"\path{figures/phase3_external_baselines/crossblock_summary_table.tex}",
            r"\path{simulation/outputs/phase3_external_baselines_full/*/summary.csv}",
            r"\path{scripts/build_paper_experiment_assets.py}; run root \path{simulation/outputs/phase3_external_baselines_full}",
        ),
        (
            r"Figures~\ref{fig:coupling-headline} and~\ref{fig:frontier-diagnostics}",
            r"\path{figures/phase3_external_baselines/block2_coupling_key.csv}; \path{frontier_key.csv}",
            r"\path{simulation/outputs/phase3_external_baselines_full/block2_coupling/summary.csv}; block summaries",
            r"\path{scripts/build_paper_experiment_assets.py}; run root \path{simulation/outputs/phase3_external_baselines_full}",
        ),
        (
            r"Appendix simulation tables",
            r"\path{figures/phase3_external_baselines/appendix_block*_full_table.tex}; \path{appendix_ablation_summary_table.tex}",
            r"\path{simulation/outputs/phase3_external_baselines_full/*/summary.csv}; \path{per_replication.csv}",
            r"\path{scripts/build_paper_experiment_assets.py}; run root \path{simulation/outputs/phase3_external_baselines_full}",
        ),
        (
            r"Table~\ref{tab:block5-precision-verification}",
            r"\path{figures/phase3_block5_precision_followup/block5_precision_table.tex}",
            r"\path{simulation/outputs/phase3_block5_precision_followup/block5_sample_size/summary.csv}",
            r"\path{scripts/build_paper_experiment_assets.py}; run root \path{simulation/outputs/phase3_block5_precision_followup}",
        ),
        (
            r"Table~\ref{tab:mlappmain}",
            r"\path{figures/application_assets/application_main_comparator_table.tex}",
            r"\path{application/outputs/tors_full_movielens_1m_rebalanced_v2_eps0p10_tau1p00/summary.csv}",
            r"\path{scripts/build_application_assets.py}; run root \path{application/outputs/tors_full_movielens_1m_rebalanced_v2_eps0p10_tau1p00}",
        ),
        (
            r"Figures~\ref{fig:app-frontier}, \ref{fig:app-policy-delta}, and~\ref{fig:app-ablation-frontier}",
            r"\path{figures/application_assets/frontier_key.csv}; \path{policy_delta_generator_shares.csv}; \path{lambda_sensitivity.csv}",
            r"\path{application/outputs/tors_full_movielens_1m_rebalanced_v2_eps0p10_tau1p00/summary.csv}; \path{policy_delta_generators.csv}",
            r"\path{scripts/build_application_assets.py}; run root \path{application/outputs/tors_full_movielens_1m_rebalanced_v2_eps0p10_tau1p00}",
        ),
        (
            r"Appendix application comparator, ablation, calibration, support, and provenance tables",
            r"\path{figures/application_assets/appendix_application_*.tex}",
            r"\path{summary.csv}; \path{logging_diagnostics.csv}; \path{dataset_provenance.csv}; processed \path{support_filter_diagnostics.csv}",
            r"\path{scripts/build_application_assets.py}; run root \path{application/outputs/tors_full_movielens_1m_rebalanced_v2_eps0p10_tau1p00}",
        ),
        (
            r"Table~\ref{tab:app-robustness-full}",
            r"\path{figures/application_assets/appendix_application_robustness_table.tex}",
            rf"\path{{{robustness_root.relative_to(ROOT)}/robustness_summary.csv}}",
            rf"\path{{application/run_tors_robustness.py}} and \path{{scripts/build_application_assets.py}}; run root \path{{{robustness_root.relative_to(ROOT)}}}",
        ),
    ]
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\scriptsize",
        r"\begin{tabularx}{\textwidth}{@{}p{0.17\textwidth}p{0.23\textwidth}p{0.25\textwidth}X@{}}",
        r"\toprule",
        r"Paper asset & Rendered artifact & Source data & Build script and run root \\",
        r"\midrule",
    ]
    for asset, artifact, source, script in rows:
        lines.append(f"{asset} & {artifact} & {source} & {script} \\\\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabularx}",
            r"\caption{Reproducibility audit for reported tables and figures. All displayed values are regenerated from saved CSV/JSON artifacts rather than transcribed manually.}",
            r"\label{tab:reproducibility-audit}",
            r"\end{table*}",
        ]
    )
    return "\n".join(lines) + "\n"


def build_manifest(robustness_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    rows = [
        ("application_main_comparator_table.tex", "Main MovieLens comparator table with burden, ESS, and max-weight diagnostics."),
        ("appendix_application_full_comparator_table.tex", "Full application comparator table with all reference rows."),
        ("appendix_application_ablation_table.tex", "Application CASP ablation table."),
        ("appendix_application_lambda_sensitivity_table.tex", "CASP lambda-sensitivity table."),
        ("appendix_application_weight_diagnostics_table.tex", "ESS and max-weight diagnostics table."),
        ("appendix_application_support_violation_table.tex", "Off-support burden-floor and generator-support diagnostic table."),
        ("appendix_application_support_table.tex", "Application support and logging diagnostics."),
        ("appendix_application_provenance_table.tex", "Application provenance and run-manifest table."),
        ("appendix_application_robustness_table.tex", "Full cached MovieLens robustness table."),
        ("appendix_reproducibility_audit_table.tex", "Reproducibility audit mapping paper assets to source data, scripts, and run roots."),
        ("frontier_key.csv", "Application value-burden frontier CSV used by pgfplots."),
        ("lambda_sensitivity.csv", "CASP lambda sensitivity CSV."),
        ("policy_delta_generator_shares.csv", "Generator-level policy delta CSV."),
        ("support_violation_diagnostics.csv", "Floor-implied off-support mass by selector."),
        ("stage1_zero_support_profile.csv", "Generator-level zero-support and selection-share profile."),
    ]
    if not robustness_rows:
        rows.append(("robustness_status", "Robustness table currently contains a pending row because no robustness_summary.csv was found."))
    return [{"artifact": artifact, "description": description} for artifact, description in rows]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build reported CASP application assets.")
    parser.add_argument("--suite-root", type=Path, default=DEFAULT_SUITE_ROOT)
    parser.add_argument("--robustness-root", type=Path, default=DEFAULT_ROBUSTNESS_ROOT)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    summary_rows = read_csv(args.suite_root / "summary.csv")
    lambda_rows = lambda_sensitivity_rows(summary_rows)
    robustness_rows = load_robustness_summary(args.robustness_root)
    zero_support_rows = stage1_zero_support_profile(args.suite_root)

    write_text(args.out_dir / "application_main_comparator_table.tex", build_main_comparator_table(summary_rows))
    write_text(args.out_dir / "appendix_application_full_comparator_table.tex", build_full_comparator_table(summary_rows))
    write_text(args.out_dir / "appendix_application_ablation_table.tex", build_ablation_table(summary_rows))
    write_text(args.out_dir / "appendix_application_lambda_sensitivity_table.tex", build_lambda_table(lambda_rows))
    write_text(args.out_dir / "appendix_application_weight_diagnostics_table.tex", build_weight_diagnostics_table(summary_rows))
    write_text(args.out_dir / "appendix_application_support_violation_table.tex", build_support_violation_table(summary_rows, zero_support_rows))
    write_text(args.out_dir / "appendix_application_support_table.tex", build_support_table(args.suite_root))
    write_text(args.out_dir / "appendix_application_provenance_table.tex", build_provenance_table(args.suite_root))
    write_text(args.out_dir / "appendix_application_robustness_table.tex", build_robustness_table(robustness_rows))
    write_text(
        args.out_dir / "appendix_reproducibility_audit_table.tex",
        build_reproducibility_audit_table(args.suite_root, args.robustness_root),
    )
    write_csv(
        args.out_dir / "frontier_key.csv",
        build_frontier_csv(summary_rows),
        ["label", "burden", "value", "stability", "ess_proxy", "max_importance_weight"],
    )
    write_csv(
        args.out_dir / "lambda_sensitivity.csv",
        lambda_rows,
        [
            "lambda",
            "label",
            "dr_value",
            "support_burden",
            "ess_proxy",
            "max_importance_weight",
            "mode_frequency",
            "unique_policies",
        ],
    )
    write_csv(
        args.out_dir / "policy_delta_generator_shares.csv",
        build_policy_delta_generator_shares(args.suite_root),
        ["generator", "dr_share", "casp_share"],
    )
    write_csv(
        args.out_dir / "support_violation_diagnostics.csv",
        offsupport_rows(summary_rows),
        ["comparator", "support_burden", "floor_implied_offsupport_mass"],
    )
    write_csv(
        args.out_dir / "stage1_zero_support_profile.csv",
        zero_support_rows,
        ["generator", "zero_stage1_support_share", "dr_selected_share", "casp_selected_share"],
    )
    write_csv(
        args.out_dir / "manifest.csv",
        build_manifest(robustness_rows),
        ["artifact", "description"],
    )


if __name__ == "__main__":
    main()
