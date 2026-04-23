from __future__ import annotations

from .utils import mean, standard_error


def expected_stage2_value(policy, env, context: tuple[float, ...], generator: int, q_model) -> float:
    pi2 = policy.stage2_probs(env, context, generator)
    total = 0.0
    for item, prob in enumerate(pi2):
        if prob <= 0.0:
            continue
        total += prob * q_model(context, generator, item)
    return total


def policy_value_true(policy, env, n_contexts: int, seed: int) -> float:
    from random import Random

    rng = Random(seed)
    values = []
    for _ in range(n_contexts):
        context = env.sample_context(rng)
        pi1 = policy.stage1_probs(env, context)
        total = 0.0
        for generator, prob1 in enumerate(pi1):
            if prob1 <= 0.0:
                continue
            pi2 = policy.stage2_probs(env, context, generator)
            for item, prob2 in enumerate(pi2):
                if prob2 <= 0.0:
                    continue
                total += prob1 * prob2 * env.reward_mean(context, generator, item)
        values.append(total)
    return mean(values)


def dr_scores(records, policy, env, reward_model) -> list[float]:
    scores = []
    for record in records:
        context = record.context
        stage1_probs = policy.stage1_probs(env, context)
        model_part = 0.0
        for generator, prob1 in enumerate(stage1_probs):
            if prob1 <= 0.0:
                continue
            stage2_probs = policy.stage2_probs(env, context, generator)
            for item, prob2 in enumerate(stage2_probs):
                if prob2 <= 0.0:
                    continue
                qhat = reward_model.predict(env.reward_features(context, generator, item))
                model_part += prob1 * prob2 * qhat

        mu1 = env.logging_stage1_probs(context)
        mu2 = env.logging_stage2_probs(context, record.stage1_action)
        pi1 = stage1_probs[record.stage1_action]
        pi2 = policy.stage2_probs(env, context, record.stage1_action)[record.stage2_action]
        if mu1[record.stage1_action] <= 0.0 or mu2[record.stage2_action] <= 0.0:
            weight = 0.0
        else:
            weight = pi1 * pi2 / (mu1[record.stage1_action] * mu2[record.stage2_action])
        qhat_obs = reward_model.predict(
            env.reward_features(context, record.stage1_action, record.stage2_action)
        )
        scores.append(model_part + weight * (record.reward - qhat_obs))
    return scores


def estimate_dr_value(records, policy, env, reward_model) -> float:
    return estimate_dr_moments(records, policy, env, reward_model)["mean"]


def estimate_dr_standard_error(records, policy, env, reward_model) -> float:
    return estimate_dr_moments(records, policy, env, reward_model)["standard_error"]


def estimate_dr_moments(records, policy, env, reward_model) -> dict[str, float]:
    scores = dr_scores(records, policy, env, reward_model)
    return {
        "mean": mean(scores),
        "standard_error": standard_error(scores),
    }


def estimate_dr_lower_confidence_bound(records, policy, env, reward_model, beta: float) -> float:
    moments = estimate_dr_moments(records, policy, env, reward_model)
    return moments["mean"] - beta * moments["standard_error"]


def importance_weights(records, policy, env) -> list[float]:
    weights = []
    for record in records:
        context = record.context
        mu1 = env.logging_stage1_probs(context)
        mu2 = env.logging_stage2_probs(context, record.stage1_action)
        pi1 = policy.stage1_probs(env, context)[record.stage1_action]
        pi2 = policy.stage2_probs(env, context, record.stage1_action)[record.stage2_action]
        weights.append(pi1 * pi2 / max(mu1[record.stage1_action] * mu2[record.stage2_action], 1e-9))
    return weights


def estimate_max_importance_weight(records, policy, env) -> float:
    weights = importance_weights(records, policy, env)
    if not weights:
        return 0.0
    return max(weights)


def estimate_ips_value(records, policy, env) -> float:
    values = []
    for record in records:
        context = record.context
        mu1 = env.logging_stage1_probs(context)
        mu2 = env.logging_stage2_probs(context, record.stage1_action)
        pi1 = policy.stage1_probs(env, context)[record.stage1_action]
        pi2 = policy.stage2_probs(env, context, record.stage1_action)[record.stage2_action]
        if mu1[record.stage1_action] <= 0.0 or mu2[record.stage2_action] <= 0.0:
            values.append(0.0)
            continue
        weight = pi1 * pi2 / (mu1[record.stage1_action] * mu2[record.stage2_action])
        values.append(weight * record.reward)
    return mean(values)


def estimate_support_burden(records, policy, env, mode: str = "full") -> float:
    burdens = []
    for record in records:
        context = record.context
        pi1 = policy.stage1_probs(env, context)
        total = 0.0
        for generator, prob1 in enumerate(pi1):
            if prob1 <= 0.0:
                continue
            mu1 = env.logging_stage1_probs(context)[generator]
            stage2_probs = policy.stage2_probs(env, context, generator)
            if mode == "stage1_only":
                total += (prob1 ** 2) / max(mu1, 1e-9)
                continue
            for item, prob2 in enumerate(stage2_probs):
                if prob2 <= 0.0:
                    continue
                mu2 = env.logging_stage2_probs(context, generator)[item]
                if mode == "stage2_only":
                    total += prob1 * (prob2 ** 2) / max(mu2, 1e-9)
                else:
                    total += (prob1 ** 2) * (prob2 ** 2) / max(mu1 * mu2, 1e-9)
        burdens.append(total)
    return mean(burdens)


def estimate_effective_sample_size(records, policy, env) -> float:
    weights_squared = [weight * weight for weight in importance_weights(records, policy, env)]
    avg = mean(weights_squared)
    if avg <= 1e-9:
        return float(len(records))
    return len(records) / avg
