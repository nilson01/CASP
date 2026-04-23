from __future__ import annotations

from dataclasses import dataclass
from random import Random

from .config import BlockConfig
from .utils import (
    argmax,
    clamp,
    dot,
    elementwise,
    mean,
    mix_with_uniform,
    one_hot,
    sample_from_probs,
    sigmoid,
    softmax,
)


@dataclass(frozen=True)
class InteractionRecord:
    context: tuple[float, ...]
    stage1_action: int
    stage2_action: int
    reward: float
    proxy: float


class SyntheticTwoStageEnvironment:
    def __init__(self, config: BlockConfig, seed: int) -> None:
        self.config = config
        self.seed = seed
        self.rng = Random(seed)
        self.generator_embeddings = [
            [self.rng.uniform(-1.0, 1.0) for _ in range(config.context_dim)]
            for _ in range(config.num_generators)
        ]
        self.item_embeddings = [
            [self.rng.uniform(-1.0, 1.0) for _ in range(config.context_dim)]
            for _ in range(config.num_items)
        ]
        self.item_proxy_bias = [self.rng.uniform(-1.0, 1.0) for _ in range(config.num_items)]
        self.item_reward_bias = [self.rng.uniform(-1.0, 1.0) for _ in range(config.num_items)]
        self.generator_logging_bias = [
            self.rng.uniform(-0.8, 0.8) for _ in range(config.num_generators)
        ]
        self.item_logging_bias = [self.rng.uniform(-0.6, 0.6) for _ in range(config.num_items)]
        self.coupling = [
            [self.rng.uniform(-1.0, 1.0) for _ in range(config.num_items)]
            for _ in range(config.num_generators)
        ]
        rare_count = int(round(config.num_items * config.rare_item_fraction))
        rare_count = min(max(rare_count, 0), config.num_items)
        self.rare_items = set(range(config.num_items - rare_count, config.num_items))

    def sample_context(self, rng: Random) -> tuple[float, ...]:
        if self.config.block_family == "counterexample":
            return (0.0,)
        return tuple(rng.uniform(-1.0, 1.0) for _ in range(self.config.context_dim))

    def feasible_set(self, context: tuple[float, ...], generator: int) -> list[int]:
        if self.config.block_family == "counterexample":
            return [generator]
        scored_items: list[tuple[float, int]] = []
        x = list(context)
        g = self.generator_embeddings[generator]
        for item in range(self.config.num_items):
            v = self.item_embeddings[item]
            retrieval_score = (
                0.45 * dot(x, v)
                + 0.35 * dot(g, v)
                + 0.20 * self.item_proxy_bias[item]
                + self.config.structured_item_scale * mean(elementwise(v, g))
            )
            scored_items.append((retrieval_score, item))
        scored_items.sort(key=lambda pair: (-pair[0], pair[1]))
        return [item for _, item in scored_items[: self.config.candidate_set_size]]

    def proxy_mean(self, context: tuple[float, ...], generator: int) -> float:
        if self.config.block_family == "counterexample":
            return 0.8 if generator == 0 else 0.2
        feasible = self.feasible_set(context, generator)
        values = []
        x = list(context)
        for item in feasible:
            v = self.item_embeddings[item]
            latent = 0.70 * self.item_proxy_bias[item] + 0.30 * dot(x, v)
            values.append(sigmoid(latent))
        return mean(values)

    def reward_mean(self, context: tuple[float, ...], generator: int, item: int) -> float:
        if self.config.block_family == "counterexample":
            if generator == 0 and item == 0:
                return 0.15
            if generator == 1 and item == 1:
                return 0.85
            return 0.0
        x = list(context)
        v = self.item_embeddings[item]
        rare_bonus = self.config.rare_item_bonus if item in self.rare_items else 0.0
        latent = (
            0.55 * self.item_reward_bias[item]
            + 0.45 * dot(x, v)
            + self.config.coupling_strength * self.coupling[generator][item]
            + rare_bonus
        )
        return sigmoid(latent)

    def logging_stage1_probs(self, context: tuple[float, ...]) -> list[float]:
        if self.config.block_family == "counterexample":
            return [0.5, 0.5]
        scores = []
        for generator in range(self.config.num_generators):
            score = (
                0.60 * self.proxy_mean(context, generator)
                + self.config.logging_skew * self.generator_logging_bias[generator]
            )
            scores.append(score)
        return mix_with_uniform(softmax(scores), self.config.min_stage1_mass)

    def logging_stage2_probs(self, context: tuple[float, ...], generator: int) -> list[float]:
        feasible = self.feasible_set(context, generator)
        scores = []
        for item in feasible:
            logging_penalty = self.config.rare_item_logging_penalty if item in self.rare_items else 0.0
            score = self.reward_mean(context, generator, item) + self.item_logging_bias[item] - logging_penalty
            scores.append(score)
        feasible_probs = mix_with_uniform(softmax(scores), self.config.min_stage2_mass)
        full_probs = [0.0] * self.config.num_items
        for item, prob in zip(feasible, feasible_probs):
            full_probs[item] = prob
        return full_probs

    def reward_features(self, context: tuple[float, ...], generator: int, item: int) -> list[float]:
        x = list(context)
        g = self.generator_embeddings[generator]
        v = self.item_embeddings[item]
        return [
            1.0,
            *x,
            *g,
            *v,
            dot(x, v),
            dot(g, v),
            dot(x, g),
            1.0 if item in self.rare_items else 0.0,
        ]

    def proxy_features(self, context: tuple[float, ...], generator: int) -> list[float]:
        x = list(context)
        g = self.generator_embeddings[generator]
        return [1.0, *x, *g, dot(x, g)]

    def sample_logged_record(self, rng: Random) -> InteractionRecord:
        context = self.sample_context(rng)
        stage1_probs = self.logging_stage1_probs(context)
        stage1_action = sample_from_probs(rng, stage1_probs)
        stage2_probs = self.logging_stage2_probs(context, stage1_action)
        stage2_action = sample_from_probs(rng, stage2_probs)
        reward_mean = self.reward_mean(context, stage1_action, stage2_action)
        reward = 1.0 if rng.random() <= reward_mean else 0.0
        proxy = clamp(
            self.proxy_mean(context, stage1_action) + rng.gauss(0.0, self.config.proxy_noise_std),
            0.0,
            1.0,
        )
        return InteractionRecord(
            context=context,
            stage1_action=stage1_action,
            stage2_action=stage2_action,
            reward=reward,
            proxy=proxy,
        )

    def sample_logged_data(self, n_records: int, seed: int) -> list[InteractionRecord]:
        rng = Random(seed)
        return [self.sample_logged_record(rng) for _ in range(n_records)]

    def oracle_stage2(self, context: tuple[float, ...], generator: int) -> int:
        feasible = self.feasible_set(context, generator)
        scores = [self.reward_mean(context, generator, item) for item in feasible]
        return feasible[argmax(scores)]

    def oracle_stage1(self, context: tuple[float, ...]) -> int:
        scores = []
        for generator in range(self.config.num_generators):
            item = self.oracle_stage2(context, generator)
            scores.append(self.reward_mean(context, generator, item))
        return argmax(scores)

