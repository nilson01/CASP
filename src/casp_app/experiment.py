from __future__ import annotations

import math
from collections import Counter, defaultdict
from pathlib import Path
from random import Random

from .config import ApplicationConfig, RunModeSpec
from .environment import (
    ReconstructedMovieLensEnvironment,
    build_logged_records,
    load_catalog,
    load_contexts,
)
from .simulation_bridge import (
    BehaviorPolicy,
    FixedGeneratorPolicy,
    GeneratorSpecificPenaltyPolicy,
    OraclePolicy,
    RandomUniformPolicy,
    RewardGreedyPolicy,
    build_stagewise_policy,
    estimate_dr_moments,
    estimate_effective_sample_size,
    estimate_ips_value,
    estimate_max_importance_weight,
    estimate_support_burden,
    fit_models_and_library,
    mean,
    policy_value_true,
    split_records,
    stddev,
)


def _candidate_burden_key(mode_spec: str) -> str:
    if mode_spec.startswith("normalized_"):
        return f"burden_{mode_spec.removeprefix('normalized_')}_norm"
    if mode_spec.startswith("raw_"):
        return f"burden_{mode_spec.removeprefix('raw_')}_raw"
    return f"burden_{mode_spec}_raw"


def _split_train_eval(records, holdout_fraction: float, seed: int):
    ordered = sorted(
        records,
        key=lambda record: (record.context.timestamp, record.context.request_id),
    )
    eval_size = max(1, int(round(len(ordered) * holdout_fraction)))
    eval_records = ordered[-eval_size:]
    train_records = ordered[:-eval_size]
    if not train_records:
        train_records = ordered[:-1]
        eval_records = ordered[-1:]
    return train_records, eval_records


def _build_candidate_diagnostics(records_select, env, reward_model, candidate_library):
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
        mode: max(sorted(values)[len(values) // 2], 1e-9)
        for mode, values in raw_burdens_by_mode.items()
    }
    for row in diagnostics:
        for mode in raw_modes:
            row[f"burden_{mode}_norm"] = row[f"burden_{mode}_raw"] / burden_scales[mode]
    return diagnostics, burden_scales


def _select_candidate(candidate_diagnostics, score_fn):
    best_row = None
    best_score = None
    for row in candidate_diagnostics:
        score = score_fn(row)
        if best_score is None or score > best_score:
            best_row = row
            best_score = score
    return best_row, best_score


def _generator_fixed_policy_diagnostics(records_select, env, reward_model, eta_grid: tuple[float, ...]):
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


def _evaluate_policy(records_eval, policy, env, reward_model, eval_contexts: int, eval_seed: int) -> dict:
    dr_moments = estimate_dr_moments(records_eval, policy, env, reward_model)
    return {
        "oracle_value": policy_value_true(policy, env, eval_contexts, eval_seed),
        "dr_value": dr_moments["mean"],
        "dr_std_error": dr_moments["standard_error"],
        "dr_lcb_1se": dr_moments["mean"] - dr_moments["standard_error"],
        "ips_value": estimate_ips_value(records_eval, policy, env),
        "support_burden": estimate_support_burden(records_eval, policy, env, mode="full"),
        "ess_proxy": estimate_effective_sample_size(records_eval, policy, env),
        "max_importance_weight": estimate_max_importance_weight(records_eval, policy, env),
    }


def _policy_stage1_choice(policy, env, context) -> int:
    probs = policy.stage1_probs(env, context)
    return max(range(len(probs)), key=lambda index: (probs[index], -index))


def _policy_stage2_choice(policy, env, context, generator: int) -> int:
    probs = policy.stage2_probs(env, context, generator)
    return max(range(len(probs)), key=lambda index: (probs[index], -index))


def _policy_delta_rows(replication: int, env, casp_policy, dr_policy) -> tuple[list[dict], list[dict]]:
    generator_pairs: Counter[tuple[int, int]] = Counter()
    item_pairs: Counter[tuple[int, int, int, int]] = Counter()
    for context in env.eval_contexts:
        dr_generator = _policy_stage1_choice(dr_policy, env, context)
        casp_generator = _policy_stage1_choice(casp_policy, env, context)
        generator_pairs[(dr_generator, casp_generator)] += 1
        dr_item = _policy_stage2_choice(dr_policy, env, context, dr_generator)
        casp_item = _policy_stage2_choice(casp_policy, env, context, casp_generator)
        item_pairs[(dr_generator, casp_generator, dr_item, casp_item)] += 1

    generator_rows = []
    total_generators = sum(generator_pairs.values()) or 1
    for (dr_generator, casp_generator), count in generator_pairs.most_common():
        generator_rows.append(
            {
                "replication": replication,
                "dr_generator": dr_generator,
                "casp_generator": casp_generator,
                "count": count,
                "share": count / total_generators,
            }
        )

    item_rows = []
    total_items = sum(item_pairs.values()) or 1
    for (dr_generator, casp_generator, dr_item, casp_item), count in item_pairs.most_common(200):
        item_rows.append(
            {
                "replication": replication,
                "dr_generator": dr_generator,
                "casp_generator": casp_generator,
                "dr_item": dr_item,
                "casp_item": casp_item,
                "count": count,
                "share": count / total_items,
            }
        )
    return generator_rows, item_rows


def _logging_diagnostic_rows(records_eval) -> list[dict]:
    support_counts = [len(record.context.support_generator_indices) for record in records_eval]
    stage1_entropies = []
    for record in records_eval:
        entropy = 0.0
        for prob in record.context.stage1_probs:
            if prob > 1e-12:
                entropy -= prob * math.log(prob)
        stage1_entropies.append(entropy)
    return [
        {"field": "eval_records", "value": len(records_eval)},
        {"field": "mean_support_generator_count", "value": mean(support_counts)},
        {"field": "max_support_generator_count", "value": max(support_counts) if support_counts else 0},
        {"field": "share_exactly_two_support_generators", "value": sum(1 for value in support_counts if value == 2) / max(len(support_counts), 1)},
        {"field": "mean_stage1_entropy", "value": mean(stage1_entropies)},
    ]


def run_single_application_split(
    processed_root: Path,
    config: ApplicationConfig,
    run_spec: RunModeSpec,
    replication_index: int,
) -> dict:
    catalog = load_catalog(processed_root)
    contexts = load_contexts(processed_root, context_cap=run_spec.context_cap)
    records = build_logged_records(contexts)
    train_pool, eval_records = _split_train_eval(records, run_spec.holdout_fraction, seed=config.seed + 17)
    records_fit, records_select = split_records(train_pool, seed=config.seed + 101 * (replication_index + 1))

    train_contexts = [record.context for record in train_pool]
    eval_contexts = [record.context for record in eval_records]
    env_train = ReconstructedMovieLensEnvironment(config, catalog, train_contexts, eval_contexts=eval_contexts)
    env_eval = ReconstructedMovieLensEnvironment(config, catalog, contexts, eval_contexts=eval_contexts)

    prepared = fit_models_and_library(records_fit, env_train, config)
    candidate_diagnostics, burden_scales = _build_candidate_diagnostics(
        records_select,
        env_train,
        prepared.reward_model,
        prepared.candidate_library,
    )

    policy_eval_seed = config.seed + 5000 + replication_index
    oracle = OraclePolicy(name="reconstructed_oracle")
    behavior = BehaviorPolicy(name="behavior")
    random_policy = RandomUniformPolicy(name="random_uniform")
    plugin_reward = RewardGreedyPolicy(
        reward_model=prepared.reward_model,
        stage1_penalty=0.0,
        stage2_penalty=0.0,
        name="plugin_reward",
    )
    stagewise = build_stagewise_policy(prepared)
    pure_value_row, pure_value_score = _select_candidate(candidate_diagnostics, lambda row: row["dr_value"])
    pure_value_policy = pure_value_row["policy"]

    fixed_generator_diagnostics = _generator_fixed_policy_diagnostics(
        records_select,
        env_train,
        prepared.reward_model,
        config.policy_eta_grid,
    )
    ma_style_stage2 = []
    for generator in range(env_train.config.num_generators):
        generator_rows = [row for row in fixed_generator_diagnostics if row["generator"] == generator]
        best_row, _ = _select_candidate(generator_rows, lambda row: row["dr_value"])
        ma_style_stage2.append(best_row["eta"])
    ma_style_policy = GeneratorSpecificPenaltyPolicy(
        reward_model=prepared.reward_model,
        stage2_penalties_by_generator=tuple(ma_style_stage2),
        name="ma_style_two_stage_opl",
    )

    wang_candidates = [row for row in fixed_generator_diagnostics if abs(row["eta"]) <= 1e-12]
    wang_row, wang_score = _select_candidate(wang_candidates, lambda row: row["dr_value"])
    wang_policy = wang_row["policy"]

    comparator_rows = []
    comparator_policies = [
        ("baseline", "reconstructed_oracle", oracle, "", 0.0),
        ("baseline", "behavior", behavior, "", 0.0),
        ("baseline", "random_uniform", random_policy, "", 0.0),
        ("baseline", "plugin_reward", plugin_reward, "", 0.0),
        ("competitor", "stagewise_proxy", stagewise, "", 0.0),
        ("competitor", "dr_value_only", pure_value_policy, pure_value_policy.name, pure_value_score),
        ("external", "ma_style_two_stage_opl", ma_style_policy, ",".join(f"{eta:.3f}" for eta in ma_style_stage2), 0.0),
        ("external", "wang_style_downstream_generator", wang_policy, wang_policy.name, wang_score),
    ]
    oracle_value = None
    for family, comparator_name, policy, selected_policy, selection_score in comparator_policies:
        metrics = _evaluate_policy(
            eval_records,
            policy,
            env_eval,
            prepared.reward_model,
            eval_contexts=run_spec.policy_eval_contexts,
            eval_seed=policy_eval_seed,
        )
        if comparator_name == "reconstructed_oracle":
            oracle_value = metrics["oracle_value"]
        comparator_rows.append(
            {
                "family": family,
                "comparator": comparator_name,
                "replication": replication_index,
                "selected_policy": selected_policy,
                "selection_score": selection_score,
                **metrics,
            }
        )

    for beta_value in config.lcb_beta_grid:
        selected_row, score = _select_candidate(
            candidate_diagnostics,
            lambda row, beta=beta_value: row["dr_value"] - beta * row["dr_std_error"],
        )
        metrics = _evaluate_policy(
            eval_records,
            selected_row["policy"],
            env_eval,
            prepared.reward_model,
            eval_contexts=run_spec.policy_eval_contexts,
            eval_seed=policy_eval_seed,
        )
        row = {
            "family": "competitor",
            "comparator": f"dr_lcb_beta_{beta_value:.2f}",
            "replication": replication_index,
            "selected_policy": selected_row["policy_name"],
            "selection_score": score,
            "selection_penalty_variant": "dr_lcb",
            **metrics,
        }
        comparator_rows.append(row)
    main_penalty_key = _candidate_burden_key("normalized_full")
    casp_policy_default = None
    for lambda_value in config.lambda_grid:
        selected_row, score = _select_candidate(
            candidate_diagnostics,
            lambda row, lam=lambda_value, key=main_penalty_key: row["dr_value"] - lam * row[key],
        )
        metrics = _evaluate_policy(
            eval_records,
            selected_row["policy"],
            env_eval,
            prepared.reward_model,
            eval_contexts=run_spec.policy_eval_contexts,
            eval_seed=policy_eval_seed,
        )
        row = {
            "family": "competitor",
            "comparator": f"casp_lambda_{lambda_value:.3f}",
            "replication": replication_index,
            "selected_policy": selected_row["policy_name"],
            "selection_score": score,
            "selection_penalty_variant": "normalized_full",
            "selection_penalty_metric": main_penalty_key,
            "selection_penalty_value": selected_row[main_penalty_key],
            "library_burden_scale_full": burden_scales["full"],
            "library_burden_scale_stage1_only": burden_scales["stage1_only"],
            "library_burden_scale_stage2_only": burden_scales["stage2_only"],
            **metrics,
        }
        comparator_rows.append(row)
        if abs(lambda_value - 0.05) <= 1e-12:
            casp_policy_default = selected_row["policy"]

    ablation_rows = []
    if run_spec.include_ablations:
        for mode in config.ablation_burden_modes:
            penalty_key = _candidate_burden_key(mode)
            selected_row, score = _select_candidate(
                candidate_diagnostics,
                lambda row, lam=config.ablation_lambda, key=penalty_key: row["dr_value"] - lam * row[key],
            )
            metrics = _evaluate_policy(
                eval_records,
                selected_row["policy"],
                env_eval,
                prepared.reward_model,
                eval_contexts=run_spec.policy_eval_contexts,
                eval_seed=policy_eval_seed,
            )
            ablation_rows.append(
                {
                    "family": "ablation",
                    "comparator": f"casp_ablation_{mode}",
                    "replication": replication_index,
                    "selected_policy": selected_row["policy_name"],
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

    all_rows = comparator_rows + ablation_rows
    if oracle_value is None:
        oracle_value = 0.0
    for row in all_rows:
        row["oracle_regret"] = oracle_value - row["oracle_value"]

    if casp_policy_default is None:
        casp_policy_default = pure_value_policy
    generator_deltas, item_deltas = _policy_delta_rows(
        replication_index,
        env_eval,
        casp_policy_default,
        pure_value_policy,
    )

    return {
        "replication": replication_index,
        "result_rows": all_rows,
        "generator_delta_rows": generator_deltas,
        "item_delta_rows": item_deltas,
        "logging_rows": _logging_diagnostic_rows(eval_records),
    }


def summarize_application_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["comparator"]].append(row)

    summary_rows = []
    for comparator, group in sorted(grouped.items()):
        record = {
            "family": group[0]["family"],
            "comparator": comparator,
            "replications": len(group),
        }
        for metric in (
            "oracle_value",
            "oracle_regret",
            "dr_value",
            "dr_std_error",
            "dr_lcb_1se",
            "ips_value",
            "support_burden",
            "ess_proxy",
            "max_importance_weight",
        ):
            values = [row[metric] for row in group]
            record[f"{metric}_mean"] = mean(values)
            record[f"{metric}_sd"] = stddev(values)
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
        summary_rows.append(record)
    return summary_rows


def selection_frequency_rows(rows: list[dict]) -> list[dict]:
    counters: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        if row.get("selected_policy"):
            counters[row["comparator"]][row["selected_policy"]] += 1
    out_rows = []
    for comparator, counts in sorted(counters.items()):
        total = sum(counts.values()) or 1
        for selected_policy, count in counts.most_common():
            out_rows.append(
                {
                    "comparator": comparator,
                    "selected_policy": selected_policy,
                    "count": count,
                    "frequency": count / total,
                }
            )
    return out_rows


def frontier_rows(summary_rows: list[dict]) -> list[dict]:
    keep = [
        "dr_value_only",
        "dr_lcb_beta_0.50",
        "casp_lambda_0.050",
        "ma_style_two_stage_opl",
    ]
    rows = []
    for row in summary_rows:
        if row["comparator"] not in keep:
            continue
        rows.append(
            {
                "comparator": row["comparator"],
                "dr_value_mean": row["dr_value_mean"],
                "support_burden_mean": row["support_burden_mean"],
                "selected_policy_mode_frequency": row["selected_policy_mode_frequency"],
            }
        )
    return rows


def aggregate_logging_rows(logging_rows_by_replication: list[list[dict]]) -> list[dict]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for rows in logging_rows_by_replication:
        for row in rows:
            grouped[row["field"]].append(float(row["value"]))
    return [{"field": field, "value": mean(values)} for field, values in sorted(grouped.items())]
