from __future__ import annotations

import csv
from pathlib import Path


PLOT_METRICS = (
    ("true_value_mean", "True Value"),
    ("oracle_regret_mean", "Oracle Regret"),
    ("dr_error_mean", "DR Estimation Error"),
    ("support_burden_mean", "Support Burden"),
    ("ess_proxy_mean", "ESS Proxy"),
    ("selected_policy_mode_frequency", "Selection Stability"),
)

KEY_COMPARATORS = (
    "oracle",
    "stagewise_proxy",
    "dr_value_only",
    "dr_lcb_beta_0.50",
    "ma_style_two_stage_opl",
    "wang_style_downstream_generator",
    "casp_lambda_0.050",
)

COMPARATOR_LABELS = {
    "oracle": "Oracle",
    "stagewise_proxy": "Stagewise proxy",
    "plugin_reward": "Plug-in reward",
    "dr_value_only": "DR value only",
    "dr_lcb_beta_0.50": "DR-LCB (beta=0.50)",
    "ma_style_two_stage_opl": "Ma-style OPL",
    "wang_style_downstream_generator": "Wang-style generator",
    "casp_lambda_0.050": "CASP (lambda=0.05)",
}

PREFERRED_ORDER = (
    "stagewise_proxy",
    "plugin_reward",
    "dr_value_only",
    "dr_lcb_beta_0.50",
    "casp_lambda_0.050",
    "ma_style_two_stage_opl",
    "wang_style_downstream_generator",
    "oracle",
)

PALETTE = (
    "#0f4c81",
    "#e67e22",
    "#2e8b57",
    "#c0392b",
    "#8e44ad",
    "#16a085",
    "#7f8c8d",
    "#d35400",
    "#1f77b4",
    "#2ca02c",
    "#9467bd",
    "#8c564b",
)


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = sorted({field for row in rows for field in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _svg_polyline(points: list[tuple[float, float]], color: str) -> str:
    if not points:
        return ""
    points_text = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    circles = "\n".join(
        f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.5" fill="{color}" />' for x, y in points
    )
    return (
        f'<polyline fill="none" stroke="{color}" stroke-width="2.2" points="{points_text}" />\n'
        f"{circles}"
    )


def _comparator_label(comparator: str) -> str:
    return COMPARATOR_LABELS.get(comparator, comparator.replace("_", " "))


def _comparator_order(comparator: str) -> tuple[int, str]:
    if comparator in PREFERRED_ORDER:
        return (PREFERRED_ORDER.index(comparator), comparator)
    return (len(PREFERRED_ORDER), comparator)


def _pick_ticks(values: list[float], max_ticks: int = 8) -> list[float]:
    unique_values = sorted(set(values))
    if len(unique_values) <= max_ticks:
        return unique_values
    if max_ticks <= 1:
        return [unique_values[0]]
    step = (len(unique_values) - 1) / (max_ticks - 1)
    selected = []
    for idx in range(max_ticks):
        picked = unique_values[round(idx * step)]
        if picked not in selected:
            selected.append(picked)
    if unique_values[-1] not in selected:
        selected[-1] = unique_values[-1]
    return selected


def _render_line_plot(
    title: str,
    y_label: str,
    grouped_rows: dict[str, list[dict]],
    metric_name: str,
    svg_path: Path,
) -> None:
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    width = 1240
    height = 720
    margin_left = 104
    margin_right = 356
    margin_top = 88
    margin_bottom = 112
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    x_values = sorted({row["sweep_value"] for rows in grouped_rows.values() for row in rows})
    y_values = [row[metric_name] for rows in grouped_rows.values() for row in rows]
    if not x_values or not y_values:
        return

    x_min = min(x_values)
    x_max = max(x_values)
    y_min = min(y_values)
    y_max = max(y_values)
    if x_min == x_max:
        x_min -= 1.0
        x_max += 1.0
    if y_min == y_max:
        padding = 1.0 if y_min == 0.0 else abs(y_min) * 0.1
        y_min -= padding
        y_max += padding
    else:
        padding = 0.1 * (y_max - y_min)
        y_min -= padding
        y_max += padding

    def x_coord(value: float) -> float:
        return margin_left + (value - x_min) / (x_max - x_min) * plot_width

    def y_coord(value: float) -> float:
        return margin_top + plot_height - (value - y_min) / (y_max - y_min) * plot_height

    grid_lines = []
    for tick in range(6):
        y_value = y_min + tick * (y_max - y_min) / 5.0
        y_pixel = y_coord(y_value)
        grid_lines.append(
            f'<line x1="{margin_left}" y1="{y_pixel:.2f}" x2="{margin_left + plot_width}" '
            f'y2="{y_pixel:.2f}" stroke="#d8dde6" stroke-width="1.1" stroke-dasharray="4 4" />'
        )
        grid_lines.append(
            f'<text x="{margin_left - 10}" y="{y_pixel + 4:.2f}" text-anchor="end" '
            f'font-size="14" font-family="Helvetica, Arial, sans-serif" fill="#374151">{y_value:.3f}</text>'
        )

    x_ticks = []
    for value in _pick_ticks(x_values, max_ticks=8):
        x_pixel = x_coord(value)
        x_ticks.append(
            f'<line x1="{x_pixel:.2f}" y1="{margin_top + plot_height}" x2="{x_pixel:.2f}" '
            f'y2="{margin_top + plot_height + 8}" stroke="#4b5563" stroke-width="1.2" />'
        )
        x_ticks.append(
            f'<text x="{x_pixel:.2f}" y="{margin_top + plot_height + 24}" text-anchor="middle" '
            f'font-size="14" font-family="Helvetica, Arial, sans-serif" fill="#374151">{value:g}</text>'
        )

    polylines = []
    legend_rows = []
    ordered_comparators = sorted(grouped_rows.keys(), key=_comparator_order)
    for index, comparator in enumerate(ordered_comparators):
        rows = sorted(grouped_rows[comparator], key=lambda row: row["sweep_value"])
        points = [(x_coord(row["sweep_value"]), y_coord(row[metric_name])) for row in rows]
        color = PALETTE[index % len(PALETTE)]
        polylines.append(_svg_polyline(points, color))
        legend_y = margin_top + 30 + 28 * index
        label = _comparator_label(comparator)
        legend_rows.append(
            f'<line x1="{width - margin_right + 20}" y1="{legend_y}" '
            f'x2="{width - margin_right + 50}" y2="{legend_y}" '
            f'stroke="{color}" stroke-width="3.0" />'
        )
        legend_rows.append(
            f'<circle cx="{width - margin_right + 35}" cy="{legend_y}" r="4.6" fill="{color}" stroke="white" stroke-width="1.1" />'
        )
        legend_rows.append(
            f'<text x="{width - margin_right + 60}" y="{legend_y + 5}" '
            f'font-size="14" font-family="Helvetica, Arial, sans-serif" fill="#111827">{label}</text>'
        )

    legend_height = min(52 + 28 * max(len(ordered_comparators), 1), plot_height + 8)

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#fcfcfd" />
<text x="{margin_left}" y="38" font-size="29" font-family="Helvetica, Arial, sans-serif" font-weight="700" fill="#111827">{title}</text>
<text x="{margin_left}" y="64" font-size="15" font-family="Helvetica, Arial, sans-serif" fill="#4b5563">{y_label} across sweep values</text>
<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_height}" stroke="#1f2937" stroke-width="1.8" />
<line x1="{margin_left}" y1="{margin_top + plot_height}" x2="{margin_left + plot_width}" y2="{margin_top + plot_height}" stroke="#1f2937" stroke-width="1.8" />
{chr(10).join(grid_lines)}
{chr(10).join(x_ticks)}
{chr(10).join(polylines)}
<text x="{margin_left + plot_width / 2:.2f}" y="{margin_top - 16}" text-anchor="middle" font-size="17" font-family="Helvetica, Arial, sans-serif" fill="#111827">Sweep value</text>
<text x="32" y="{margin_top + plot_height / 2:.2f}" text-anchor="middle" font-size="17" font-family="Helvetica, Arial, sans-serif" fill="#111827" transform="rotate(-90 32 {margin_top + plot_height / 2:.2f})">{y_label}</text>
<rect x="{width - margin_right}" y="{margin_top}" rx="10" ry="10" width="{margin_right - 28}" height="{legend_height:.2f}" fill="#f3f4f6" stroke="#d1d5db" stroke-width="1.1" />
<text x="{width - margin_right + 20}" y="{margin_top + 18}" font-size="14" font-family="Helvetica, Arial, sans-serif" font-weight="700" fill="#111827">Comparator</text>
{chr(10).join(legend_rows)}
</svg>
"""
    svg_path.write_text(svg, encoding="utf-8")


def _plot_rows(summary_rows: list[dict], metric_name: str, key_only: bool) -> list[dict]:
    rows = []
    for row in summary_rows:
        comparator = row["comparator"]
        if key_only and comparator not in KEY_COMPARATORS:
            continue
        if metric_name not in row:
            continue
        rows.append(
            {
                "block": row["block"],
                "family": row["family"],
                "comparator": comparator,
                "sweep_value": row["sweep_value"],
                "metric": metric_name,
                "value": row[metric_name],
                "sd": row.get(metric_name.replace("_mean", "_sd"), 0.0),
            }
        )
    return rows


def generate_block_plots(block_output_dir: Path, summary_rows: list[dict]) -> list[dict]:
    manifests = []
    plots_dir = block_output_dir / "plots"
    plot_data_dir = block_output_dir / "plot_data"
    for metric_name, label in PLOT_METRICS:
        for view_name, key_only in (("all", False), ("key", True)):
            rows = _plot_rows(summary_rows, metric_name, key_only=key_only)
            if not rows:
                continue
            csv_path = plot_data_dir / f"{view_name}__{metric_name}.csv"
            svg_path = plots_dir / f"{view_name}__{metric_name}.svg"
            _write_csv(csv_path, rows)
            grouped: dict[str, list[dict]] = {}
            for row in rows:
                grouped.setdefault(row["comparator"], []).append(row)
            _render_line_plot(
                title=f"{label} ({view_name} comparators)",
                y_label=label,
                grouped_rows=grouped,
                metric_name="value",
                svg_path=svg_path,
            )
            manifests.append(
                {
                    "metric": metric_name,
                    "view": view_name,
                    "plot_file": str(svg_path),
                    "data_file": str(csv_path),
                }
            )
    return manifests
