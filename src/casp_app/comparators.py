from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ComparatorSpec:
    name: str
    label: str
    family: str
    selection_target: str
    burden_role: str
    paper_role: str

    def to_dict(self) -> dict:
        return asdict(self)


def application_comparators() -> list[ComparatorSpec]:
    return [
        ComparatorSpec(
            name="stagewise_proxy",
            label="Stagewise proxy",
            family="competitor",
            selection_target="proxy-driven generator choice plus feasible-set reward greedy stage 2",
            burden_role="none",
            paper_role="main comparator",
        ),
        ComparatorSpec(
            name="plugin_reward",
            label="Plug-in reward",
            family="baseline",
            selection_target="reward-model greedy end-to-end policy",
            burden_role="none",
            paper_role="baseline",
        ),
        ComparatorSpec(
            name="dr_value_only",
            label="DR value only",
            family="competitor",
            selection_target="maximize doubly robust value over a shared policy library",
            burden_role="none",
            paper_role="main comparator",
        ),
        ComparatorSpec(
            name="dr_lcb_beta_0.50",
            label="DR-LCB 0.50",
            family="competitor",
            selection_target="maximize a generic conservative DR lower-confidence score",
            burden_role="generic uncertainty",
            paper_role="main comparator",
        ),
        ComparatorSpec(
            name="casp_lambda_0.050",
            label="CASP 0.05",
            family="competitor",
            selection_target="maximize DR value minus support burden",
            burden_role="stage-induced feasible support",
            paper_role="main comparator",
        ),
        ComparatorSpec(
            name="ma_style_two_stage_opl",
            label="Ma-style OPL",
            family="external",
            selection_target="two-stage OPL-style end-to-end learner over the shared application library",
            burden_role="implicit only",
            paper_role="closest external baseline",
        ),
        ComparatorSpec(
            name="wang_style_downstream_generator",
            label="Wang-style generator",
            family="external",
            selection_target="generator-only downstream-aware selector with fixed best response",
            burden_role="none",
            paper_role="stage-1 downstream-aware reference",
        ),
        ComparatorSpec(
            name="behavior",
            label="Behavior",
            family="baseline",
            selection_target="reconstructed logging policy",
            burden_role="n/a",
            paper_role="reference only",
        ),
        ComparatorSpec(
            name="random_uniform",
            label="Random uniform",
            family="baseline",
            selection_target="uniform over generators and feasible items",
            burden_role="n/a",
            paper_role="reference only",
        ),
        ComparatorSpec(
            name="reconstructed_oracle",
            label="Reconstructed oracle",
            family="baseline",
            selection_target="debug-only upper reference under the semi-synthetic reward surface",
            burden_role="n/a",
            paper_role="debug only",
        ),
    ]
