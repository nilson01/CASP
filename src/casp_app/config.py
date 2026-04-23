from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class GeneratorSpec:
    generator_id: str
    label: str
    description: str
    score_formula: str
    support_effect: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class FigureTableSpec:
    asset_id: str
    asset_type: str
    manuscript_location: str
    title: str
    source_path_hint: str
    note: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class RunModeSpec:
    mode: str
    label: str
    description: str
    context_cap: int
    split_replications: int
    holdout_fraction: float
    policy_eval_contexts: int
    include_ablations: bool

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ApplicationConfig:
    name: str
    dataset_id: str
    seed: int
    raw_archive_name: str
    raw_download_url: str
    raw_data_subdir: str
    min_history_length: int
    positive_rating_threshold: float
    candidate_set_size: int
    context_limit: int
    min_supporting_generators: int
    fallback_min_supporting_generators: int
    fallback_context_floor: int
    fallback_pool_strategy: str
    fallback_head_generator_index: int
    fallback_singleton_caps: tuple[int, ...]
    collaborative_neighbor_strategy: str
    holdout_fraction: float
    stage1_temperature: float
    stage1_exploration_epsilon: float
    stage2_temperature: float
    min_stage1_mass: float
    min_stage2_mass: float
    ridge_alpha: float
    policy_kappa_grid: tuple[float, ...]
    policy_eta_grid: tuple[float, ...]
    lambda_grid: tuple[float, ...]
    lcb_beta_grid: tuple[float, ...]
    ablation_lambda: float
    ablation_burden_modes: tuple[str, ...]
    smoke_context_limit: int
    smoke_replications: int
    full_replications: int
    policy_eval_contexts: int
    generators: tuple[GeneratorSpec, ...]

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["generators"] = [generator.to_dict() for generator in self.generators]
        payload["run_modes"] = [spec.to_dict() for spec in default_run_mode_specs(self).values()]
        return payload


def default_output_root() -> Path:
    return Path(__file__).resolve().parents[2] / "outputs" / "runs" / "application"


def default_raw_root() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "raw"


def default_processed_root() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "processed"


def default_run_mode_specs(config: ApplicationConfig) -> dict[str, RunModeSpec]:
    return {
        "smoke": RunModeSpec(
            mode="smoke",
            label="Smoke run",
            description="Tiny end-to-end execution on about 2,000 chronologically prepared contexts with a temporal holdout, the full comparator family, ablations, manifests, and policy-delta outputs.",
            context_cap=config.smoke_context_limit,
            split_replications=config.smoke_replications,
            holdout_fraction=config.holdout_fraction,
            policy_eval_contexts=min(1200, config.policy_eval_contexts),
            include_ablations=True,
        ),
        "full": RunModeSpec(
            mode="full",
            label="Full run",
            description="Paper-facing application run on up to 25,000 chronologically prepared contexts with a fixed temporal holdout, 20 replicated selection splits inside the earlier training pool, appendix assets, and policy-delta diagnostics.",
            context_cap=config.context_limit,
            split_replications=config.full_replications,
            holdout_fraction=config.holdout_fraction,
            policy_eval_contexts=config.policy_eval_contexts,
            include_ablations=True,
        ),
    }


def run_mode_spec(config: ApplicationConfig, mode: str) -> RunModeSpec:
    specs = default_run_mode_specs(config)
    if mode not in specs:
        raise KeyError(f"Unknown run mode: {mode}")
    return specs[mode]


DEFAULT_APPLICATION_CONFIGS = {
    "movielens_1m_reconstructed": ApplicationConfig(
        name="movielens_1m_reconstructed",
        dataset_id="ml-1m",
        seed=20260411,
        raw_archive_name="ml-1m.zip",
        raw_download_url="https://files.grouplens.org/datasets/movielens/ml-1m.zip",
        raw_data_subdir="ml-1m",
        min_history_length=5,
        positive_rating_threshold=4.0,
        candidate_set_size=30,
        context_limit=25000,
        min_supporting_generators=2,
        fallback_min_supporting_generators=1,
        fallback_context_floor=10000,
        fallback_pool_strategy="singleton_diversity_preserving_cap_g1_v1",
        fallback_head_generator_index=0,
        fallback_singleton_caps=(25000, 3300, 25000, 25000),
        collaborative_neighbor_strategy="hierarchical_age_sex_backoff_v1",
        holdout_fraction=0.20,
        stage1_temperature=0.55,
        stage1_exploration_epsilon=0.05,
        stage2_temperature=0.45,
        min_stage1_mass=0.05,
        min_stage2_mass=0.03,
        ridge_alpha=1.0,
        policy_kappa_grid=(0.0, 0.02, 0.04, 0.08, 0.12),
        policy_eta_grid=(0.0, 0.02, 0.04, 0.08, 0.12),
        lambda_grid=(0.0, 0.01, 0.02, 0.05, 0.10, 0.20),
        lcb_beta_grid=(0.5, 1.0, 1.5, 2.0),
        ablation_lambda=0.05,
        ablation_burden_modes=(
            "normalized_full",
            "normalized_stage1_only",
            "normalized_stage2_only",
            "raw_full",
        ),
        smoke_context_limit=2000,
        smoke_replications=3,
        full_replications=20,
        policy_eval_contexts=2500,
        generators=(
            GeneratorSpec(
                generator_id="popularity_head",
                label="Popularity head",
                description="Head-heavy retriever favoring globally frequent items.",
                score_formula="0.75 * (0.70 * popularity + 0.20 * mean rating + 0.10 * genre affinity) + 0.25 * request reranker",
                support_effect="High coverage on popular items and weak long-tail exploration.",
            ),
            GeneratorSpec(
                generator_id="genre_match",
                label="Genre match",
                description="Retriever aligned to the user's positive-genre profile.",
                score_formula="0.90 * (0.75 * genre affinity + 0.15 * mean rating + 0.10 * novelty) + 0.10 * request reranker",
                support_effect="Improves topical relevance while narrowing feasible support.",
            ),
            GeneratorSpec(
                generator_id="collaborative_neighbor",
                label="Collaborative neighbor",
                description="Retriever based on positive feedback from similar demographic slices.",
                score_formula="0.85 * (0.55 * hierarchical collaborative backoff + 0.20 * genre affinity + 0.15 * mean rating + 0.10 * popularity) + 0.15 * request reranker",
                support_effect="Can expose niche items with uneven logger support.",
            ),
            GeneratorSpec(
                generator_id="long_tail_explorer",
                label="Long-tail explorer",
                description="Retriever that trades off affinity against popularity concentration.",
                score_formula="0.90 * (0.55 * genre affinity + 0.35 * novelty + 0.10 * mean rating) + 0.10 * request reranker",
                support_effect="Most likely to expose weak-support but potentially high-value items.",
            ),
        ),
    )
}
