from __future__ import annotations

from dataclasses import dataclass

from .models import RidgeRegressor
from .utils import argmax, one_hot


class Policy:
    name: str

    def stage1_probs(self, env, context: tuple[float, ...]) -> list[float]:
        raise NotImplementedError

    def stage2_probs(self, env, context: tuple[float, ...], generator: int) -> list[float]:
        raise NotImplementedError


def best_item_under_penalty(
    reward_model: RidgeRegressor,
    env,
    context: tuple[float, ...],
    generator: int,
    stage2_penalty: float,
) -> int:
    feasible = env.feasible_set(context, generator)
    mu2 = env.logging_stage2_probs(context, generator)
    scores = []
    for item in feasible:
        qhat = reward_model.predict(env.reward_features(context, generator, item))
        score = qhat - stage2_penalty / max(mu2[item], 1e-9)
        scores.append(score)
    return feasible[argmax(scores)]


@dataclass
class BehaviorPolicy(Policy):
    name: str = "behavior"

    def stage1_probs(self, env, context: tuple[float, ...]) -> list[float]:
        return env.logging_stage1_probs(context)

    def stage2_probs(self, env, context: tuple[float, ...], generator: int) -> list[float]:
        return env.logging_stage2_probs(context, generator)


@dataclass
class RandomUniformPolicy(Policy):
    name: str = "random_uniform"

    def stage1_probs(self, env, context: tuple[float, ...]) -> list[float]:
        weight = 1.0 / env.config.num_generators
        return [weight] * env.config.num_generators

    def stage2_probs(self, env, context: tuple[float, ...], generator: int) -> list[float]:
        feasible = env.feasible_set(context, generator)
        full_probs = [0.0] * env.config.num_items
        weight = 1.0 / len(feasible)
        for item in feasible:
            full_probs[item] = weight
        return full_probs


@dataclass
class OraclePolicy(Policy):
    name: str = "oracle"

    def stage1_probs(self, env, context: tuple[float, ...]) -> list[float]:
        return one_hot(env.config.num_generators, env.oracle_stage1(context))

    def stage2_probs(self, env, context: tuple[float, ...], generator: int) -> list[float]:
        return one_hot(env.config.num_items, env.oracle_stage2(context, generator))


@dataclass
class RewardGreedyPolicy(Policy):
    reward_model: RidgeRegressor
    stage1_penalty: float
    stage2_penalty: float
    name: str

    def _best_item(self, env, context: tuple[float, ...], generator: int) -> int:
        return best_item_under_penalty(
            reward_model=self.reward_model,
            env=env,
            context=context,
            generator=generator,
            stage2_penalty=self.stage2_penalty,
        )

    def _best_generator(self, env, context: tuple[float, ...]) -> int:
        mu1 = env.logging_stage1_probs(context)
        scores = []
        for generator in range(env.config.num_generators):
            best_item = self._best_item(env, context, generator)
            qhat = self.reward_model.predict(env.reward_features(context, generator, best_item))
            score = qhat - self.stage1_penalty / max(mu1[generator], 1e-9)
            scores.append(score)
        return argmax(scores)

    def stage1_probs(self, env, context: tuple[float, ...]) -> list[float]:
        return one_hot(env.config.num_generators, self._best_generator(env, context))

    def stage2_probs(self, env, context: tuple[float, ...], generator: int) -> list[float]:
        return one_hot(env.config.num_items, self._best_item(env, context, generator))


@dataclass
class FixedGeneratorPolicy(Policy):
    reward_model: RidgeRegressor
    generator: int
    stage2_penalty: float
    name: str

    def stage1_probs(self, env, context: tuple[float, ...]) -> list[float]:
        return one_hot(env.config.num_generators, self.generator)

    def stage2_probs(self, env, context: tuple[float, ...], generator: int) -> list[float]:
        best_item = best_item_under_penalty(
            reward_model=self.reward_model,
            env=env,
            context=context,
            generator=generator,
            stage2_penalty=self.stage2_penalty,
        )
        return one_hot(env.config.num_items, best_item)


@dataclass
class GeneratorSpecificPenaltyPolicy(Policy):
    reward_model: RidgeRegressor
    stage2_penalties_by_generator: tuple[float, ...]
    name: str

    def _best_item(self, env, context: tuple[float, ...], generator: int) -> int:
        return best_item_under_penalty(
            reward_model=self.reward_model,
            env=env,
            context=context,
            generator=generator,
            stage2_penalty=self.stage2_penalties_by_generator[generator],
        )

    def stage1_probs(self, env, context: tuple[float, ...]) -> list[float]:
        scores = []
        for generator in range(env.config.num_generators):
            best_item = self._best_item(env, context, generator)
            qhat = self.reward_model.predict(env.reward_features(context, generator, best_item))
            scores.append(qhat)
        return one_hot(env.config.num_generators, argmax(scores))

    def stage2_probs(self, env, context: tuple[float, ...], generator: int) -> list[float]:
        return one_hot(env.config.num_items, self._best_item(env, context, generator))


@dataclass
class StagewiseProxyPolicy(Policy):
    proxy_model: RidgeRegressor
    reward_model: RidgeRegressor
    name: str = "stagewise_proxy"

    def stage1_probs(self, env, context: tuple[float, ...]) -> list[float]:
        scores = [
            self.proxy_model.predict(env.proxy_features(context, generator))
            for generator in range(env.config.num_generators)
        ]
        return one_hot(env.config.num_generators, argmax(scores))

    def stage2_probs(self, env, context: tuple[float, ...], generator: int) -> list[float]:
        feasible = env.feasible_set(context, generator)
        scores = [
            self.reward_model.predict(env.reward_features(context, generator, item))
            for item in feasible
        ]
        return one_hot(env.config.num_items, feasible[argmax(scores)])


def build_candidate_library(
    env,
    reward_model: RidgeRegressor,
    kappa_grid: tuple[float, ...],
    eta_grid: tuple[float, ...],
) -> list[RewardGreedyPolicy]:
    policies: list[RewardGreedyPolicy] = []
    for stage1_penalty in kappa_grid:
        for stage2_penalty in eta_grid:
            name = f"candidate_k{stage1_penalty:.3f}_e{stage2_penalty:.3f}"
            policies.append(
                RewardGreedyPolicy(
                    reward_model=reward_model,
                    stage1_penalty=stage1_penalty,
                    stage2_penalty=stage2_penalty,
                    name=name,
                )
            )
    return policies
