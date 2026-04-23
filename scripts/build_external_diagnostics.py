from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


ROOT = Path(__file__).resolve().parents[1]
SUITE_ROOT = ROOT / "outputs" / "runs" / "simulation" / "phase3_external_baselines_full"
OUT_DIR = ROOT / "outputs" / "figures" / "simulation"

BLOCKS = [
    ("block2_coupling", "B2"),
    ("block3_support", "B3"),
    ("block4_large_action", "B4"),
    ("block5_sample_size", "B5"),
]

COMPARATORS = [
    ("casp_lambda_0.050", "CASP 0.05", "#1f77b4"),
    ("dr_lcb_beta_0.50", "DR-LCB 0.50", "#2ca02c"),
    ("dr_value_only", "DR value only", "#ff7f0e"),
    ("ma_style_two_stage_opl", "Ma-style OPL", "#d62728"),
]

COMPARATOR_INDEX = {comparator: idx for idx, (comparator, _, _) in enumerate(COMPARATORS)}


def mean(rows: Iterable[float]) -> float:
    values = list(rows)
    return sum(values) / len(values) if values else 0.0


def load_block_averages() -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for block_name, block_label in BLOCKS:
        summary_path = SUITE_ROOT / block_name / "summary.csv"
        with summary_path.open() as handle:
            rows = list(csv.DictReader(handle))
        for comparator_key, comparator_label, color in COMPARATORS:
            subset = [row for row in rows if row["comparator"] == comparator_key]
            records.append(
                {
                    "block": block_name,
                    "block_label": block_label,
                    "comparator": comparator_key,
                    "comparator_label": comparator_label,
                    "color": color,
                    "avg_value": mean(float(row["true_value_mean"]) for row in subset),
                    "avg_regret": mean(float(row["oracle_regret_mean"]) for row in subset),
                    "avg_burden": mean(float(row["support_burden_mean"]) for row in subset),
                    "avg_stability": mean(
                        float(row["selected_policy_mode_frequency"] or 0.0)
                        for row in subset
                    ),
                }
            )
    return records


def write_csv(path: Path, rows: List[Dict[str, object]], fields: List[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


def scale(value: float, lo: float, hi: float, out_lo: float, out_hi: float) -> float:
    if hi <= lo:
        return (out_lo + out_hi) / 2.0
    frac = (value - lo) / (hi - lo)
    return out_lo + frac * (out_hi - out_lo)


def build_scatter_svg(
    title: str,
    x_label: str,
    y_label: str,
    points: List[Dict[str, object]],
    x_key: str,
    y_key: str,
    out_path: Path,
) -> None:
    width = 1240
    height = 760
    left = 108
    right = 330
    top = 96
    bottom = 126
    plot_w = width - left - right
    plot_h = height - top - bottom

    xs = [float(point[x_key]) for point in points]
    ys = [float(point[y_key]) for point in points]
    x_min = min(xs)
    x_max = max(xs)
    y_min = min(ys)
    y_max = max(ys)
    x_pad = max((x_max - x_min) * 0.08, 1.0)
    y_pad = max((y_max - y_min) * 0.10, 0.01)
    x_lo = x_min - x_pad
    x_hi = x_max + x_pad
    y_lo = max(0.0, y_min - y_pad)
    y_hi = y_max + y_pad

    def x_pos(value: float) -> float:
        return scale(value, x_lo, x_hi, left, left + plot_w)

    def y_pos(value: float) -> float:
        return scale(value, y_lo, y_hi, top + plot_h, top)

    x_ticks = 5
    y_ticks = 5
    block_legend = "B2=coupling, B3=support, B4=large action, B5=sample size"

    svg: List[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fcfcfd"/>',
        f'<text x="{width/2:.1f}" y="46" text-anchor="middle" font-size="31" font-family="Helvetica, Arial, sans-serif" font-weight="700" fill="#111827">{title}</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#1f2937" stroke-width="2"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#1f2937" stroke-width="2"/>',
    ]

    for idx in range(x_ticks + 1):
        value = x_lo + (x_hi - x_lo) * idx / x_ticks
        pos = x_pos(value)
        svg.append(
            f'<line x1="{pos:.1f}" y1="{top + plot_h}" x2="{pos:.1f}" y2="{top + plot_h + 8}" stroke="#4b5563" stroke-width="1.2"/>'
        )
        svg.append(
            f'<text x="{pos:.1f}" y="{top + plot_h + 32}" text-anchor="middle" font-size="14" font-family="Helvetica, Arial, sans-serif" fill="#374151">{value:.1f}</text>'
        )

    for idx in range(y_ticks + 1):
        value = y_lo + (y_hi - y_lo) * idx / y_ticks
        pos = y_pos(value)
        svg.append(
            f'<line x1="{left - 8}" y1="{pos:.1f}" x2="{left}" y2="{pos:.1f}" stroke="#4b5563" stroke-width="1.2"/>'
        )
        svg.append(
            f'<text x="{left - 14}" y="{pos + 4:.1f}" text-anchor="end" font-size="14" font-family="Helvetica, Arial, sans-serif" fill="#374151">{value:.3f}</text>'
        )
        svg.append(
            f'<line x1="{left}" y1="{pos:.1f}" x2="{left + plot_w}" y2="{pos:.1f}" stroke="#d8dde6" stroke-width="1.1" stroke-dasharray="4 4"/>'
        )

    for point in points:
        cx = x_pos(float(point[x_key]))
        cy = y_pos(float(point[y_key]))
        color = str(point["color"])
        label = f"{point['block_label']}"
        comparator_key = str(point["comparator"])
        idx = COMPARATOR_INDEX.get(comparator_key, 0)
        x_shift = 11 if idx in (0, 2) else -11
        y_shift = -9 if idx in (0, 1) else 16
        anchor = "start" if x_shift > 0 else "end"
        svg.append(
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="7.4" fill="{color}" stroke="white" stroke-width="1.5"/>'
        )
        svg.append(
            f'<text x="{cx + x_shift:.1f}" y="{cy + y_shift:.1f}" text-anchor="{anchor}" font-size="12.8" font-family="Helvetica, Arial, sans-serif" fill="{color}" font-weight="600">{label}</text>'
        )

    legend_x = left + plot_w + 36
    legend_y = top + 34
    svg.append(
        f'<rect x="{legend_x - 18}" y="{legend_y - 28}" width="266" height="150" rx="10" ry="10" fill="#f3f4f6" stroke="#d1d5db" stroke-width="1.1"/>'
    )
    svg.append(
        f'<text x="{legend_x}" y="{legend_y - 6}" font-size="14" font-family="Helvetica, Arial, sans-serif" fill="#111827" font-weight="700">Comparator</text>'
    )
    for idx, (_, label, color) in enumerate(COMPARATORS):
        y = legend_y + 24 + idx * 28
        svg.append(f'<circle cx="{legend_x}" cy="{y}" r="6.4" fill="{color}" stroke="white" stroke-width="1"/>')
        svg.append(
            f'<text x="{legend_x + 16}" y="{y + 4}" font-size="14" font-family="Helvetica, Arial, sans-serif" fill="#111827">{label}</text>'
        )

    svg.extend(
        [
            f'<text x="{left + plot_w / 2:.1f}" y="{top - 16}" text-anchor="middle" font-size="18" font-family="Helvetica, Arial, sans-serif" fill="#111827">{x_label}</text>',
            f'<text x="32" y="{top + plot_h / 2:.1f}" text-anchor="middle" font-size="18" font-family="Helvetica, Arial, sans-serif" fill="#111827" transform="rotate(-90 32 {top + plot_h / 2:.1f})">{y_label}</text>',
            f'<text x="{left}" y="{height - 18}" font-size="13" font-family="Helvetica, Arial, sans-serif" fill="#4b5563">{block_legend}</text>',
            "</svg>",
        ]
    )

    out_path.write_text("\n".join(svg))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records = load_block_averages()

    value_burden_csv = OUT_DIR / "value_burden_summary.csv"
    stability_burden_csv = OUT_DIR / "stability_burden_summary.csv"
    manifest_csv = OUT_DIR / "manifest.csv"

    write_csv(
        value_burden_csv,
        records,
        [
            "block",
            "block_label",
            "comparator",
            "comparator_label",
            "avg_value",
            "avg_regret",
            "avg_burden",
        ],
    )
    write_csv(
        stability_burden_csv,
        records,
        [
            "block",
            "block_label",
            "comparator",
            "comparator_label",
            "avg_stability",
            "avg_burden",
            "avg_value",
            "avg_regret",
        ],
    )

    value_svg = OUT_DIR / "value_burden_frontier.svg"
    stability_svg = OUT_DIR / "stability_burden_frontier.svg"

    build_scatter_svg(
        title="Phase 3 External Suite: Value vs Support Burden",
        x_label="Average support burden",
        y_label="Average true policy value",
        points=records,
        x_key="avg_burden",
        y_key="avg_value",
        out_path=value_svg,
    )
    build_scatter_svg(
        title="Phase 3 External Suite: Stability vs Support Burden",
        x_label="Average support burden",
        y_label="Average selected-policy mode frequency",
        points=records,
        x_key="avg_burden",
        y_key="avg_stability",
        out_path=stability_svg,
    )

    manifest_rows = [
        {
            "artifact": "value_burden_summary.csv",
            "path": str(value_burden_csv.relative_to(ROOT)),
            "description": "Underlying data for the value-versus-burden diagnostic.",
        },
        {
            "artifact": "value_burden_frontier.svg",
            "path": str(value_svg.relative_to(ROOT)),
            "description": "Scatter plot of average value versus average burden across Blocks 2-5.",
        },
        {
            "artifact": "stability_burden_summary.csv",
            "path": str(stability_burden_csv.relative_to(ROOT)),
            "description": "Underlying data for the stability-versus-burden diagnostic.",
        },
        {
            "artifact": "stability_burden_frontier.svg",
            "path": str(stability_svg.relative_to(ROOT)),
            "description": "Scatter plot of average selection stability versus average burden across Blocks 2-5.",
        },
    ]

    with manifest_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["artifact", "path", "description"])
        writer.writeheader()
        writer.writerows(manifest_rows)


if __name__ == "__main__":
    main()
