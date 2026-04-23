from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from .config import BlockConfig, with_sweep_value
from .dgp import SyntheticTwoStageEnvironment
from .estimators import (
    estimate_dr_moments,
    estimate_effective_sample_size,
    estimate_ips_value,
    estimate_max_importance_weight,
    estimate_support_burden,
    policy_value_true,
)
from .learners import (
    build_stagewise_policy,
    fit_models_and_library,
    split_records,
)
from .policies import (
    BehaviorPolicy,
    FixedGeneratorPolicy,
    GeneratorSpecificPenaltyPolicy,
    OraclePolicy,
    RandomUniformPolicy,
    RewardGreedyPolicy,
)
from .utils import mean, median, stddev


def evaluate_policy(records_select, policy, env, reward_model, eval_contexts: int, eval_seed: int) -> dict:
    true_value = policy_value_true(policy, env, eval_contexts, eval_seed)
    dr_moments = estimate_dr_moments(records_select, policy, env, reward_model)
    dr_value = dr_moments["mean"]
    dr_std_error = dr_moments["standard_error"]
    return {
        "true_value": true_value,
        "dr_value": dr_value,
        "dr_error": dr_value - true_value,
        "dr_std_error": dr_std_error,
        "dr_lcb_1se": dr_value - dr_std_error,
        "ips_value": estimate_ips_value(records_select, policy, env),
        "support_burden": estimate_support_burden(records_select, policy, env, mode="full"),
        "ess_proxy": estimate_effective_sample_size(records_select, policy, env),
        "max_importance_weight": estimate_max_importance_weight(records_select, policy, env),
    }


def _candidate_burden_key(mode_spec: str, default_variant: str) -> str:
    if mode_spec.startswith("normalized_"):
        return f"burden_{mode_spec.removeprefix('normalized_')}_norm"
    if mode_spec.startswith("raw_"):
        return f"burden_{mode_spec.removeprefix('raw_')}_raw"
    suffix = "norm" if default_variant == "library_median" else "raw"
    return f"burden_{mode_spec}_{suffix}"


def build_candidate_diagnostics(records_select, env, reward_model, candidate_library) -> tuple[list[dict], dict[str, float]]:
    diagnostics = []
    raw_modes = ("full", "stage1_only", "stage2_only")
    raw_burdens_by_mode = {mode: [] for mode in raw_modes}
    for index, policy in enumerate(candidate_library):
        dr_moments = estimate_dr_moments(records_select, policy, env, reward_model)
        row = {
            "index": index,
            "policy": policy,
            "policy_name": policy.name,
            "dr_value": dr_moments["mean"],
            "dr_std_error": dr_moments["standard_error"],
        }
        for mode in raw_modes:
            raw_value = estimate_support_burden(records_select, policy, env, mode=mode)
            row[f"burden_{mode}_raw"] = raw_value
            raw_burdens_by_mode[mode].append(raw_value)
        diagnostics.append(row)

    burden_scales = {
        mode: max(median(values), 1e-9)
        for mode, values in raw_burdens_by_mode.items()
    }
    for row in diagnostics:
        for mode in raw_modes:
            row[f"burden_{mode}_norm"] = row[f"burden_{mode}_raw"] / burden_scales[mode]
    return diagnostics, burden_scales


def select_candidate(candidate_diagnostics: list[dict], score_fn) -> tuple[dict, float]:
    best_row = None
    best_score = None
    for row in candidate_diagnostics:
        score = score_fn(row)
        if best_score is None or score > best_score:
            best_score = score
            best_row = row
    return best_row, best_score


def generator_fixed_policy_diagnostics(records_select, env, reward_model, eta_grid: tuple[float, ...]) -> list[dict]:
    diagnostics = []
    for generator in range(env.config.num_generators):
        for eta in eta_grid:
            policy = FixedGeneratorPolicy(
                reward_model=reward_model,
                generator=generator,
                stage2_penalty=eta,
                name=f"fixed_g{generator}_e{eta:.3f}",
            )
            dr_moments = estimate_dr_moments(records_select, policy, env, reward_model)
            diagnostics.append(
                {
                    "generator": generator,
                    "eta": eta,
                    "policy": policy,
                    "policy_name": policy.name,
                    "dr_value": dr_moments["mean"],
                    "dr_std_error": dr_moments["standard_error"],
                }
            )
    return diagnostics


def run_single_replication(config: BlockConfig, sweep_value: float, replication_index: int) -> list[dict]:
    replication_seed = config.seed + 1000 * replication_index + int(100 * sweep_value)
    effective_config = with_sweep_value(config, sweep_value)
    if effective_config.sweep_name == "num_items":
        effective_config = BlockConfig(**{**effective_config.to_dict(), "num_items": int(round(sweep_value))})

    env = SyntheticTwoStageEnvironment(effective_config, seed=replication_seed)
    records = env.sample_logged_data(effective_config.train_size, seed=replication_seed + 11)
    records_fit, records_select = split_records(records, seed=replication_seed + 17)
    prepared = fit_models_and_library(records_fit, env, effective_config)
    candidate_diagnostics, burden_scales = build_candidate_diagnostics(
        records_select,
        env,
        prepared.reward_model,
        prepared.candidate_library,
    )
    policy_eval_seed = replication_seed + 29

    results = []
    oracle = OraclePolicy()
    behavior = BehaviorPolicy()
    random_policy = RandomUniformPolicy()
    plugin_reward = RewardGreedyPolicy(
        reward_model=prepared.reward_model,
        stage1_penalty=0.0,
        stage2_penalty=0.0,
        name="plugin_reward",
    )
    stagewise = build_stagewise_policy(prepared)
    pure_value_row, _ = select_candidate(candidate_diagnostics, lambda row: row["dr_value"])
    pure_value_policy = pure_value_row["policy"]
    fixed_generator_diagnostics = generator_fixed_policy_diagnostics(
        records_select,
        env,
        prepared.reward_model,
        effective_config.policy_eta_grid,
    )

    ma_style_stage2 = []
    for generator in range(env.config.num_generators):
        generator_rows = [row for row in fixed_generator_diagnostics if row["generator"] == generator]
        best_row, _ = select_candidate(generator_rows, lambda row: row["dr_value"])
        ma_style_stage2.append(best_row["eta"])
    ma_style_policy = GeneratorSpecificPenaltyPolicy(
        reward_model=prepared.reward_model,
        stage2_penalties_by_generator=tuple(ma_style_stage2),
        name="ma_style_two_stage_opl",
    )

    wang_candidates = [
        row
        for row in fixed_generator_diagnostics
        if abs(row["eta"]) <= 1e-12
    ]
    wang_row, wang_score = select_candidate(wang_candidates, lambda row: row["dr_value"])
    wang_policy = wang_row["policy"]

    comparator_policies = [
        ("baseline", "oracle", oracle, ""),
        ("baseline", "behavior", behavior, ""),
        ("baseline", "random_uniform", random_policy, ""),
        ("baseline", "plugin_reward", plugin_reward, ""),
        ("competitor", "stagewise_proxy", stagewise, ""),
        ("competitor", "dr_value_only", pure_value_policy, pure_value_policy.name),
        ("external", "ma_style_two_stage_opl", ma_style_policy, ",".join(f"{eta:.3f}" for eta in ma_style_stage2)),
        ("external", "wang_style_downstream_generator", wang_policy, wang_policy.name),
    ]

    oracle_true_value = None
    for family, comparator_name, policy, selected_policy in comparator_policies:
        metrics = evaluate_policy(
            records_select,
            policy,
            env,
            prepared.reward_model,
            eval_contexts=effective_config.eval_contexts,
            eval_seed=policy_eval_seed,
        )
        if comparator_name == "oracle":
            oracle_true_value = metrics["true_value"]
        row = {
            "block": effective_config.name,
            "family": family,
            "comparator": comparator_name,
            "selected_policy": selected_policy,
            "replication": replication_index,
            "sweep_name": effective_config.sweep_name,
            "sweep_value": sweep_value,
            **metrics,
        }
        if comparator_name == "ma_style_two_stage_opl":
            row["selection_score"] = estimate_dr_moments(records_select, policy, env, prepared.reward_model)["mean"]
            row["selection_penalty_variant"] = "per_generator_stage2_dr"
        if comparator_name == "wang_style_downstream_generator":
            row["selection_score"] = wang_score
            row["selection_penalty_variant"] = "generator_only_downstream_dr"
        results.append(row)

    for beta_value in effective_config.lcb_beta_grid:
        selected_row, score = select_candidate(
            candidate_diagnostics,
            lambda row, beta=beta_value: row["dr_value"] - beta * row["dr_std_error"],
        )
        policy = selected_row["policy"]
        metrics = evaluate_policy(
            records_select,
            policy,
            env,
            prepared.reward_model,
            eval_contexts=effective_config.eval_contexts,
            eval_seed=policy_eval_seed,
        )
        results.append(
            {
                "block": effective_config.name,
                "family": "competitor",
                "comparator": f"dr_lcb_beta_{beta_value:.2f}",
                "selected_policy": selected_row["policy_name"],
                "replication": replication_index,
                "sweep_name": effective_config.sweep_name,
                "sweep_value": sweep_value,
                "selection_score": score,
                "selection_penalty_variant": "dr_lcb",
                **metrics,
            }
        )

    main_penalty_key = _candidate_burden_key("full", effective_config.casp_penalty_variant)
    for lambda_value in effective_config.lambda_grid:
        selected_row, score = select_candidate(
            candidate_diagnostics,
            lambda row, lam=lambda_value, key=main_penalty_key: row["dr_value"] - lam * row[key],
        )
        policy = selected_row["policy"]
        metrics = evaluate_policy(
            records_select,
            policy,
            env,
            prepared.reward_model,
            eval_contexts=effective_config.eval_contexts,
            eval_seed=policy_eval_seed,
        )
        results.append(
            {
                "block": effective_config.name,
                "family": "competitor",
                "comparator": f"casp_lambda_{lambda_value:.3f}",
                "selected_policy": selected_row["policy_name"],
                "replication": replication_index,
                "sweep_name": effective_config.sweep_name,
                "sweep_value": sweep_value,
                "selection_score": score,
                "selection_penalty_variant": effective_config.casp_penalty_variant,
                "selection_penalty_metric": main_penalty_key,
                "selection_penalty_value": selected_row[main_penalty_key],
                "library_burden_scale_full": burden_scales["full"],
                "library_burden_scale_stage1_only": burden_scales["stage1_only"],
                "library_burden_scale_stage2_only": burden_scales["stage2_only"],
                **metrics,
            }
        )

    for mode in effective_config.ablation_burden_modes:
        penalty_key = _candidate_burden_key(mode, effective_config.casp_penalty_variant)
        selected_row, score = select_candidate(
            candidate_diagnostics,
            lambda row, lam=effective_config.ablation_lambda, key=penalty_key: row["dr_value"] - lam * row[key],
        )
        policy = selected_row["policy"]
        metrics = evaluate_policy(
            records_select,
            policy,
            env,
            prepared.reward_model,
            eval_contexts=effective_config.eval_contexts,
            eval_seed=policy_eval_seed,
        )
        results.append(
            {
                "block": effective_config.name,
                "family": "ablation",
                "comparator": f"casp_ablation_{mode}",
                "selected_policy": selected_row["policy_name"],
                "replication": replication_index,
                "sweep_name": effective_config.sweep_name,
                "sweep_value": sweep_value,
                "selection_score": score,
                "selection_penalty_variant": mode,
                "selection_penalty_metric": penalty_key,
                "selection_penalty_value": selected_row[penalty_key],
                "library_burden_scale_full": burden_scales["full"],
                "library_burden_scale_stage1_only": burden_scales["stage1_only"],
                "library_burden_scale_stage2_only": burden_scales["stage2_only"],
                **metrics,
            }
        )

    if oracle_true_value is None:
        oracle_true_value = 0.0
    for row in results:
        row["oracle_regret"] = oracle_true_value - row["true_value"]

    return results


def summarize_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, float], list[dict]] = {}
    for row in rows:
        key = (row["family"], row["comparator"], row["sweep_value"])
        grouped.setdefault(key, []).append(row)

    summary = []
    metric_names = [
        "true_value",
        "oracle_regret",
        "dr_value",
        "dr_error",
        "dr_std_error",
        "dr_lcb_1se",
        "ips_value",
        "support_burden",
        "ess_proxy",
        "max_importance_weight",
    ]
    for (family, comparator, sweep_value), group in sorted(grouped.items()):
        record = {
            "block": group[0]["block"],
            "family": family,
            "comparator": comparator,
            "sweep_value": sweep_value,
            "replications": len(group),
        }
        for metric in metric_names:
            values = [row[metric] for row in group]
            record[f"{metric}_mean"] = mean(values)
            record[f"{metric}_sd"] = stddev(values)
        selection_scores = [row["selection_score"] for row in group if "selection_score" in row]
        if selection_scores:
            record["selection_score_mean"] = mean(selection_scores)
            record["selection_score_sd"] = stddev(selection_scores)
        selected = [row["selected_policy"] for row in group if row.get("selected_policy")]
        if selected:
            counts = Counter(selected)
            top_policy, top_count = counts.most_common(1)[0]
            record["selected_policy_mode"] = top_policy
            record["selected_policy_mode_frequency"] = top_count / len(selected)
            record["selected_policy_unique_count"] = len(counts)
        else:
            record["selected_policy_mode"] = ""
            record["selected_policy_mode_frequency"] = 0.0
            record["selected_policy_unique_count"] = 0
        summary.append(record)
    return summary


def selection_frequency_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, float], list[str]] = {}
    for row in rows:
        selected_policy = row.get("selected_policy")
        if not selected_policy:
            continue
        key = (row["block"], row["comparator"], row["sweep_value"])
        grouped.setdefault(key, []).append(selected_policy)

    output = []
    for (block, comparator, sweep_value), selected in sorted(grouped.items()):
        counts = Counter(selected)
        total = len(selected)
        for policy_name, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
            output.append(
                {
                    "block": block,
                    "comparator": comparator,
                    "sweep_value": sweep_value,
                    "selected_policy": policy_name,
                    "count": count,
                    "frequency": count / total,
                }
            )
    return output


def write_rows_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = sorted({field for row in rows for field in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def run_block(config: BlockConfig, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for sweep_value in config.sweep_values:
        for replication_index in range(config.replications):
            all_rows.extend(run_single_replication(config, sweep_value, replication_index))

    summary_rows = summarize_rows(all_rows)
    selection_rows = selection_frequency_rows(all_rows)
    write_rows_csv(output_dir / "per_replication.csv", all_rows)
    write_rows_csv(output_dir / "summary.csv", summary_rows)
    write_rows_csv(output_dir / "selection_frequency.csv", selection_rows)
    write_json(output_dir / "summary.json", summary_rows)
    write_json(output_dir / "selection_frequency.json", selection_rows)
    write_json(output_dir / "config.json", config.to_dict())
    return {
        "config": config.to_dict(),
        "rows": all_rows,
        "summary": summary_rows,
        "selection_frequency": selection_rows,
    }
