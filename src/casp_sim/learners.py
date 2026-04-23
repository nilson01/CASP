from __future__ import annotations

from dataclasses import dataclass
from random import Random

from .estimators import estimate_dr_lower_confidence_bound, estimate_dr_value, estimate_support_burden
from .models import RidgeRegressor
from .policies import StagewiseProxyPolicy, build_candidate_library


@dataclass
class PreparedModels:
    reward_model: RidgeRegressor
    proxy_model: RidgeRegressor
    candidate_library: list


def split_records(records, seed: int) -> tuple[list, list]:
    rng = Random(seed)
    shuffled = records[:]
    rng.shuffle(shuffled)
    midpoint = len(shuffled) // 2
    return shuffled[:midpoint], shuffled[midpoint:]


def fit_models_and_library(records_fit, env, config) -> PreparedModels:
    reward_features = [
        env.reward_features(record.context, record.stage1_action, record.stage2_action)
        for record in records_fit
    ]
    reward_targets = [record.reward for record in records_fit]
    reward_model = RidgeRegressor.fit(
        reward_features,
        reward_targets,
        alpha=config.ridge_alpha,
        clip_min=0.0,
        clip_max=1.0,
    )

    proxy_features = [env.proxy_features(record.context, record.stage1_action) for record in records_fit]
    proxy_targets = [record.proxy for record in records_fit]
    proxy_model = RidgeRegressor.fit(
        proxy_features,
        proxy_targets,
        alpha=config.ridge_alpha,
        clip_min=0.0,
        clip_max=1.0,
    )

    library = build_candidate_library(
        env,
        reward_model=reward_model,
        kappa_grid=config.policy_kappa_grid,
        eta_grid=config.policy_eta_grid,
    )
    return PreparedModels(reward_model=reward_model, proxy_model=proxy_model, candidate_library=library)


def build_stagewise_policy(prepared: PreparedModels) -> StagewiseProxyPolicy:
    return StagewiseProxyPolicy(
        proxy_model=prepared.proxy_model,
        reward_model=prepared.reward_model,
    )


def select_by_value(records_select, env, reward_model, candidate_library):
    best_policy = None
    best_score = None
    for policy in candidate_library:
        score = estimate_dr_value(records_select, policy, env, reward_model)
        if best_score is None or score > best_score:
            best_score = score
            best_policy = policy
    return best_policy, best_score


def select_by_pessimism(records_select, env, reward_model, candidate_library, lambda_value: float, mode: str):
    best_policy = None
    best_score = None
    for policy in candidate_library:
        dr_score = estimate_dr_value(records_select, policy, env, reward_model)
        burden = estimate_support_burden(records_select, policy, env, mode=mode)
        score = dr_score - lambda_value * burden
        if best_score is None or score > best_score:
            best_score = score
            best_policy = policy
    return best_policy, best_score


def select_by_dr_lcb(records_select, env, reward_model, candidate_library, beta: float):
    best_policy = None
    best_score = None
    for policy in candidate_library:
        score = estimate_dr_lower_confidence_bound(
            records_select,
            policy,
            env,
            reward_model,
            beta=beta,
        )
        if best_score is None or score > best_score:
            best_score = score
            best_policy = policy
    return best_policy, best_score
