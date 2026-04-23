from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path


@dataclass(frozen=True)
class BlockConfig:
    name: str
    seed: int
    block_family: str
    replications: int
    train_size: int
    eval_contexts: int
    context_dim: int
    num_generators: int
    num_items: int
    candidate_set_size: int
    ridge_alpha: float
    proxy_noise_std: float
    min_stage1_mass: float
    min_stage2_mass: float
    coupling_strength: float
    logging_skew: float
    rare_item_fraction: float
    rare_item_bonus: float
    rare_item_logging_penalty: float
    structured_item_scale: float
    policy_kappa_grid: tuple[float, ...]
    policy_eta_grid: tuple[float, ...]
    lambda_grid: tuple[float, ...]
    lcb_beta_grid: tuple[float, ...]
    ablation_lambda: float
    ablation_burden_modes: tuple[str, ...]
    sweep_name: str
    sweep_values: tuple[float, ...]
    casp_penalty_variant: str = "raw"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class FigureTableSpec:
    asset_id: str
    asset_type: str
    manuscript_location: str
    label_hint: str
    title: str
    block: str
    source_kind: str
    view: str = ""
    metric: str = ""
    source_table: str = ""
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def with_sweep_value(config: BlockConfig, value: float) -> BlockConfig:
    if not config.sweep_name:
        return config
    int_fields = {"num_items", "train_size", "eval_contexts", "candidate_set_size", "num_generators"}
    if config.sweep_name in int_fields:
        value = int(round(value))
    return replace(config, **{config.sweep_name: value})


def default_output_root() -> Path:
    return Path(__file__).resolve().parents[2] / "outputs" / "runs" / "simulation"


DEFAULT_BLOCKS = {
    "block1_counterexample": BlockConfig(
        name="block1_counterexample",
        seed=20260401,
        block_family="counterexample",
        replications=12,
        train_size=300,
        eval_contexts=600,
        context_dim=1,
        num_generators=2,
        num_items=2,
        candidate_set_size=1,
        ridge_alpha=1.0,
        proxy_noise_std=0.02,
        min_stage1_mass=0.10,
        min_stage2_mass=0.10,
        coupling_strength=0.0,
        logging_skew=0.0,
        rare_item_fraction=0.0,
        rare_item_bonus=0.0,
        rare_item_logging_penalty=0.0,
        structured_item_scale=0.0,
        policy_kappa_grid=(0.0, 0.02, 0.05, 0.10),
        policy_eta_grid=(0.0, 0.02, 0.05, 0.10),
        lambda_grid=(0.0, 0.01, 0.02, 0.05, 0.10),
        lcb_beta_grid=(0.5, 1.0, 1.5),
        ablation_lambda=0.05,
        ablation_burden_modes=("full", "stage1_only", "stage2_only"),
        sweep_name="coupling_strength",
        sweep_values=(0.0,),
    ),
    "block2_coupling": BlockConfig(
        name="block2_coupling",
        seed=20260402,
        block_family="generic",
        replications=36,
        train_size=900,
        eval_contexts=1800,
        context_dim=3,
        num_generators=4,
        num_items=24,
        candidate_set_size=4,
        ridge_alpha=1.0,
        proxy_noise_std=0.05,
        min_stage1_mass=0.08,
        min_stage2_mass=0.05,
        coupling_strength=0.4,
        logging_skew=0.2,
        rare_item_fraction=0.10,
        rare_item_bonus=0.0,
        rare_item_logging_penalty=0.0,
        structured_item_scale=0.30,
        policy_kappa_grid=(0.0, 0.02, 0.04, 0.08, 0.12),
        policy_eta_grid=(0.0, 0.02, 0.04, 0.08, 0.12),
        lambda_grid=(0.0, 0.01, 0.02, 0.05, 0.10, 0.20),
        lcb_beta_grid=(0.5, 1.0, 1.5, 2.0),
        ablation_lambda=0.05,
        ablation_burden_modes=("full", "stage1_only", "stage2_only"),
        sweep_name="coupling_strength",
        sweep_values=(0.0, 0.25, 0.50, 0.75, 1.00, 1.50, 2.00),
    ),
    "block3_support": BlockConfig(
        name="block3_support",
        seed=20260403,
        block_family="generic",
        replications=36,
        train_size=900,
        eval_contexts=1800,
        context_dim=3,
        num_generators=4,
        num_items=24,
        candidate_set_size=4,
        ridge_alpha=1.0,
        proxy_noise_std=0.05,
        min_stage1_mass=0.05,
        min_stage2_mass=0.03,
        coupling_strength=0.8,
        logging_skew=0.3,
        rare_item_fraction=0.20,
        rare_item_bonus=1.25,
        rare_item_logging_penalty=1.10,
        structured_item_scale=0.35,
        policy_kappa_grid=(0.0, 0.02, 0.04, 0.08, 0.12),
        policy_eta_grid=(0.0, 0.02, 0.04, 0.08, 0.12),
        lambda_grid=(0.0, 0.01, 0.02, 0.05, 0.10, 0.20),
        lcb_beta_grid=(0.5, 1.0, 1.5, 2.0),
        ablation_lambda=0.05,
        ablation_burden_modes=("full", "stage1_only", "stage2_only"),
        sweep_name="logging_skew",
        sweep_values=(0.0, 0.4, 0.8, 1.2, 1.6, 2.0, 2.4, 3.0),
    ),
    "block4_large_action": BlockConfig(
        name="block4_large_action",
        seed=20260404,
        block_family="generic",
        replications=24,
        train_size=1200,
        eval_contexts=1800,
        context_dim=4,
        num_generators=6,
        num_items=80,
        candidate_set_size=8,
        ridge_alpha=1.5,
        proxy_noise_std=0.05,
        min_stage1_mass=0.07,
        min_stage2_mass=0.03,
        coupling_strength=0.9,
        logging_skew=0.8,
        rare_item_fraction=0.15,
        rare_item_bonus=0.8,
        rare_item_logging_penalty=0.8,
        structured_item_scale=0.55,
        policy_kappa_grid=(0.0, 0.02, 0.04, 0.08, 0.12),
        policy_eta_grid=(0.0, 0.02, 0.04, 0.08, 0.12),
        lambda_grid=(0.0, 0.01, 0.02, 0.05, 0.10, 0.20),
        lcb_beta_grid=(0.5, 1.0, 1.5, 2.0),
        ablation_lambda=0.05,
        ablation_burden_modes=("full", "stage1_only", "stage2_only"),
        sweep_name="num_items",
        sweep_values=(40.0, 80.0, 120.0, 160.0),
    ),
    "block5_sample_size": BlockConfig(
        name="block5_sample_size",
        seed=20260405,
        block_family="generic",
        replications=24,
        train_size=800,
        eval_contexts=1800,
        context_dim=3,
        num_generators=4,
        num_items=24,
        candidate_set_size=4,
        ridge_alpha=1.0,
        proxy_noise_std=0.05,
        min_stage1_mass=0.05,
        min_stage2_mass=0.03,
        coupling_strength=0.8,
        logging_skew=1.2,
        rare_item_fraction=0.15,
        rare_item_bonus=1.0,
        rare_item_logging_penalty=0.9,
        structured_item_scale=0.35,
        policy_kappa_grid=(0.0, 0.02, 0.04, 0.08, 0.12),
        policy_eta_grid=(0.0, 0.02, 0.04, 0.08, 0.12),
        lambda_grid=(0.0, 0.01, 0.02, 0.05, 0.10, 0.20),
        lcb_beta_grid=(0.5, 1.0, 1.5, 2.0),
        ablation_lambda=0.05,
        ablation_burden_modes=("full", "stage1_only", "stage2_only"),
        sweep_name="train_size",
        sweep_values=(150.0, 300.0, 600.0, 1200.0, 2400.0, 4800.0),
    ),
}


def _phase2_block(config: BlockConfig) -> BlockConfig:
    return replace(
        config,
        policy_kappa_grid=(0.0, 0.01, 0.02, 0.04, 0.08, 0.12),
        policy_eta_grid=(0.0, 0.01, 0.02, 0.04, 0.08, 0.12),
        lambda_grid=(0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75),
        lcb_beta_grid=(0.5, 1.0),
        ablation_lambda=0.20,
        ablation_burden_modes=(
            "normalized_full",
            "normalized_stage1_only",
            "normalized_stage2_only",
            "raw_full",
        ),
        casp_penalty_variant="library_median",
    )


PHASE2_BLOCKS = {
    block_name: _phase2_block(config)
    for block_name, config in DEFAULT_BLOCKS.items()
}


BLOCK5_PRECISION_BLOCKS = {
    "block5_sample_size": replace(
        PHASE2_BLOCKS["block5_sample_size"],
        replications=48,
        sweep_values=(600.0, 1200.0, 2400.0, 4800.0),
    )
}


BLOCK_SETS = {
    "phase1": DEFAULT_BLOCKS,
    "phase2": PHASE2_BLOCKS,
    "phase2_block5_precision": BLOCK5_PRECISION_BLOCKS,
}


LOCKED_FIGURE_TABLE_SPECS = (
    FigureTableSpec(
        asset_id="tbl_main_counterexample",
        asset_type="table",
        manuscript_location="main",
        label_hint="Table 1",
        title="Minimal counterexample summary",
        block="block1_counterexample",
        source_kind="summary_table",
        source_table="summary.csv",
        note="Restrict to sweep_value=0.0 and report oracle, stagewise_proxy, dr_value_only, generic DR-LCB, and CASP comparators.",
    ),
    FigureTableSpec(
        asset_id="fig_main_coupling_value",
        asset_type="figure",
        manuscript_location="main",
        label_hint="Figure 1",
        title="Coupling sweep: true policy value",
        block="block2_coupling",
        source_kind="plot",
        view="key",
        metric="true_value_mean",
        note="Headline comparison showing how end-to-end value changes as stage coupling strengthens.",
    ),
    FigureTableSpec(
        asset_id="fig_main_support_regret",
        asset_type="figure",
        manuscript_location="main",
        label_hint="Figure 2",
        title="Support stress: oracle regret",
        block="block3_support",
        source_kind="plot",
        view="key",
        metric="oracle_regret_mean",
        note="Main deployment-facing regret comparison under worsening support.",
    ),
    FigureTableSpec(
        asset_id="fig_main_support_diagnostic",
        asset_type="figure",
        manuscript_location="main",
        label_hint="Figure 3",
        title="Support stress: burden diagnostic",
        block="block3_support",
        source_kind="plot",
        view="key",
        metric="support_burden_mean",
        note="Diagnostic figure showing whether CASP responds to support stress in the intended way.",
    ),
    FigureTableSpec(
        asset_id="fig_main_large_action",
        asset_type="figure",
        manuscript_location="main",
        label_hint="Figure 4",
        title="Large-action stress: true policy value",
        block="block4_large_action",
        source_kind="plot",
        view="key",
        metric="true_value_mean",
        note="Large-item-universe comparison for the key competitor family.",
    ),
    FigureTableSpec(
        asset_id="fig_main_sample_size",
        asset_type="figure",
        manuscript_location="main",
        label_hint="Figure 5",
        title="Sample-size sweep: oracle regret",
        block="block5_sample_size",
        source_kind="plot",
        view="key",
        metric="oracle_regret_mean",
        note="Data-efficiency plot showing how quickly conservative selection stabilizes.",
    ),
    FigureTableSpec(
        asset_id="tbl_main_selection_stability",
        asset_type="table",
        manuscript_location="main",
        label_hint="Table 2",
        title="Selection stability summary",
        block="block5_sample_size",
        source_kind="summary_table",
        source_table="summary.csv",
        note="Use selected_policy_mode_frequency and selected_policy_unique_count columns to summarize stability by sample size.",
    ),
    FigureTableSpec(
        asset_id="fig_appx_coupling_all",
        asset_type="figure",
        manuscript_location="appendix",
        label_hint="Figure A1",
        title="Coupling sweep: all-comparator true value",
        block="block2_coupling",
        source_kind="plot",
        view="all",
        metric="true_value_mean",
        note="Appendix expansion with all comparators shown.",
    ),
    FigureTableSpec(
        asset_id="fig_appx_support_stability",
        asset_type="figure",
        manuscript_location="appendix",
        label_hint="Figure A2",
        title="Support stress: all-comparator selection stability",
        block="block3_support",
        source_kind="plot",
        view="all",
        metric="selected_policy_mode_frequency",
        note="Appendix diagnostic for selection concentration across competitors.",
    ),
    FigureTableSpec(
        asset_id="fig_appx_large_action_regret",
        asset_type="figure",
        manuscript_location="appendix",
        label_hint="Figure A3",
        title="Large-action stress: all-comparator oracle regret",
        block="block4_large_action",
        source_kind="plot",
        view="all",
        metric="oracle_regret_mean",
        note="Appendix view emphasizing regret rather than absolute value.",
    ),
    FigureTableSpec(
        asset_id="fig_appx_sample_size_error",
        asset_type="figure",
        manuscript_location="appendix",
        label_hint="Figure A4",
        title="Sample-size sweep: all-comparator DR estimation error",
        block="block5_sample_size",
        source_kind="plot",
        view="all",
        metric="dr_error_mean",
        note="Appendix plot emphasizing estimation error rather than policy value.",
    ),
)


def locked_sweep_grid_rows(blocks: dict[str, BlockConfig] | None = None) -> list[dict]:
    active_blocks = blocks or DEFAULT_BLOCKS
    rows = []
    for block_name, config in active_blocks.items():
        for sweep_value in config.sweep_values:
            rows.append(
                {
                    "block": block_name,
                    "sweep_name": config.sweep_name,
                    "sweep_value": sweep_value,
                    "replications": config.replications,
                    "train_size": config.train_size,
                    "eval_contexts": config.eval_contexts,
                    "context_dim": config.context_dim,
                    "num_generators": config.num_generators,
                    "num_items": config.num_items,
                    "candidate_set_size": config.candidate_set_size,
                    "lambda_grid": config.lambda_grid,
                    "lcb_beta_grid": config.lcb_beta_grid,
                    "casp_penalty_variant": config.casp_penalty_variant,
                }
            )
    return rows


def materialize_figure_table_map(output_root: Path) -> list[dict]:
    rows = []
    for spec in LOCKED_FIGURE_TABLE_SPECS:
        row = spec.to_dict()
        block_root = output_root / spec.block
        if spec.source_kind == "plot":
            row["plot_file"] = str(block_root / "plots" / f"{spec.view}__{spec.metric}.svg")
            row["plot_data_file"] = str(block_root / "plot_data" / f"{spec.view}__{spec.metric}.csv")
        else:
            row["table_file"] = str(block_root / spec.source_table)
        rows.append(row)
    return rows
