from __future__ import annotations

import argparse
from dataclasses import replace

from casp_app.config import DEFAULT_APPLICATION_CONFIGS, default_output_root
from casp_app.pipeline import (
    build_application_foundation,
    prepare_application_data,
    run_application_mode,
    validate_application_mode,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the CASP application pipeline.")
    parser.add_argument(
        "--mode",
        default="foundation",
        choices=["foundation", "validate", "prepare-data", "smoke", "full"],
        help="Build the foundation, validate run configuration, prepare data, or execute smoke/full later.",
    )
    parser.add_argument(
        "--dataset",
        default="movielens_1m_reconstructed",
        choices=sorted(DEFAULT_APPLICATION_CONFIGS.keys()),
        help="Locked application dataset configuration.",
    )
    parser.add_argument(
        "--suite-name",
        default="application_movielens_1m",
        help="Output subdirectory name under outputs/runs/application.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun existing task files for smoke/full execution modes.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate smoke/full configuration without touching dataset artifacts.",
    )
    parser.add_argument(
        "--config-name-override",
        default=None,
        help="Optional processed-dataset/config name override, useful for selected rebalanced variants.",
    )
    parser.add_argument(
        "--stage1-temperature",
        type=float,
        default=None,
        help="Optional override for the stage-1 softmax temperature recorded in run artifacts.",
    )
    parser.add_argument(
        "--stage1-exploration-epsilon",
        type=float,
        default=None,
        help="Optional override for the explicit stage-1 exploration mixture recorded in run artifacts.",
    )
    parser.add_argument(
        "--candidate-set-size",
        type=int,
        default=None,
        help="Optional override for the reconstructed top-L feasible set size.",
    )
    parser.add_argument(
        "--positive-rating-threshold",
        type=float,
        default=None,
        help="Optional override for the binary reward threshold.",
    )
    parser.add_argument(
        "--min-stage1-mass",
        type=float,
        default=None,
        help="Optional override for the legacy stage-1 minimum mass field recorded in artifacts.",
    )
    parser.add_argument(
        "--min-stage2-mass",
        type=float,
        default=None,
        help="Optional override for the stage-2 minimum mass floor.",
    )
    parser.add_argument(
        "--context-limit",
        type=int,
        default=None,
        help="Optional override for the prepared context cap.",
    )
    parser.add_argument(
        "--smoke-replications",
        type=int,
        default=None,
        help="Optional override for smoke split replications.",
    )
    parser.add_argument(
        "--full-replications",
        type=int,
        default=None,
        help="Optional override for full split replications.",
    )
    parser.add_argument(
        "--policy-eval-contexts",
        type=int,
        default=None,
        help="Optional override for reconstructed-oracle evaluation context count.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = DEFAULT_APPLICATION_CONFIGS[args.dataset]
    if (
        args.config_name_override is not None
        or args.stage1_temperature is not None
        or args.stage1_exploration_epsilon is not None
        or args.candidate_set_size is not None
        or args.positive_rating_threshold is not None
        or args.min_stage1_mass is not None
        or args.min_stage2_mass is not None
        or args.context_limit is not None
        or args.smoke_replications is not None
        or args.full_replications is not None
        or args.policy_eval_contexts is not None
    ):
        config = replace(
            config,
            name=args.config_name_override or config.name,
            stage1_temperature=args.stage1_temperature
            if args.stage1_temperature is not None
            else config.stage1_temperature,
            stage1_exploration_epsilon=args.stage1_exploration_epsilon
            if args.stage1_exploration_epsilon is not None
            else config.stage1_exploration_epsilon,
            candidate_set_size=args.candidate_set_size
            if args.candidate_set_size is not None
            else config.candidate_set_size,
            positive_rating_threshold=args.positive_rating_threshold
            if args.positive_rating_threshold is not None
            else config.positive_rating_threshold,
            min_stage1_mass=args.min_stage1_mass
            if args.min_stage1_mass is not None
            else config.min_stage1_mass,
            min_stage2_mass=args.min_stage2_mass
            if args.min_stage2_mass is not None
            else config.min_stage2_mass,
            context_limit=args.context_limit
            if args.context_limit is not None
            else config.context_limit,
            smoke_replications=args.smoke_replications
            if args.smoke_replications is not None
            else config.smoke_replications,
            full_replications=args.full_replications
            if args.full_replications is not None
            else config.full_replications,
            policy_eval_contexts=args.policy_eval_contexts
            if args.policy_eval_contexts is not None
            else config.policy_eval_contexts,
        )
    output_root = default_output_root() / args.suite_name

    if args.mode == "foundation":
        suite_info = build_application_foundation(output_root=output_root, config=config)
    elif args.mode == "validate":
        suite_info = validate_application_mode(output_root=output_root, config=config, mode="smoke")
        full_suite_info = validate_application_mode(
            output_root=default_output_root() / f"{args.suite_name}_full_validation",
            config=config,
            mode="full",
        )
        print(f"Smoke validation status: {suite_info['status']}")
        print(f"Full validation status: {full_suite_info['status']}")
        return
    elif args.mode == "prepare-data":
        suite_info = prepare_application_data(output_root=output_root, config=config)
    else:
        suite_info = run_application_mode(
            output_root=output_root,
            config=config,
            mode=args.mode,
            force=args.force,
            dry_run=args.dry_run,
        )

    print(f"Application artifacts are in {output_root}")
    print(f"Status: {suite_info['status']}")


if __name__ == "__main__":
    main()
