from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from random import Random

from .config import ApplicationConfig


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        exp_term = math.exp(-value)
        return 1.0 / (1.0 + exp_term)
    exp_term = math.exp(value)
    return exp_term / (1.0 + exp_term)


@dataclass(frozen=True)
class ApplicationEnvConfig:
    num_generators: int
    num_items: int


@dataclass(frozen=True)
class CatalogItem:
    item_index: int
    movie_id: int
    title: str
    release_year: int | None
    genres: tuple[str, ...]
    rating_count: int
    mean_rating: float
    positive_rate: float
    popularity_norm: float
    novelty_score: float


@dataclass(frozen=True)
class ApplicationContext:
    request_id: int
    user_id: int
    timestamp: int
    history_length: int
    positive_history_length: int
    recent_mean_rating: float
    user_sex: str
    user_age: int
    user_occupation: str
    top_genres: tuple[str, ...]
    positive_genre_counter: dict[str, int]
    observed_item_index: int
    observed_reward: int
    stage1_action: int
    stage2_action: int
    support_generator_indices: tuple[int, ...]
    stage1_probs: tuple[float, ...]
    generator_scores: tuple[float, ...]
    proxy_values: tuple[float, ...]
    candidate_sets: tuple[tuple[int, ...], ...]
    stage2_probs: tuple[tuple[float, ...], ...]
    seen_movie_count: int


@dataclass(frozen=True)
class ApplicationLoggedRecord:
    context: ApplicationContext
    stage1_action: int
    stage2_action: int
    reward: float
    proxy: float


def load_catalog(processed_root: Path) -> list[CatalogItem]:
    path = processed_root / "catalog.csv"
    with path.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            rows.append(
                CatalogItem(
                    item_index=int(row["item_index"]),
                    movie_id=int(row["movie_id"]),
                    title=row["title"],
                    release_year=int(row["release_year"]) if row["release_year"] else None,
                    genres=tuple(token for token in row["genres"].split("|") if token),
                    rating_count=int(row["rating_count"]),
                    mean_rating=float(row["mean_rating"]),
                    positive_rate=float(row["positive_rate"]),
                    popularity_norm=float(row["popularity_norm"]),
                    novelty_score=float(row["novelty_score"]),
                )
            )
    return rows


def load_contexts(processed_root: Path, context_cap: int | None = None) -> list[ApplicationContext]:
    path = processed_root / "request_contexts.csv"
    contexts: list[ApplicationContext] = []
    with path.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            contexts.append(
                ApplicationContext(
                    request_id=int(row["request_id"]),
                    user_id=int(row["user_id"]),
                    timestamp=int(row["timestamp"]),
                    history_length=int(row["history_length"]),
                    positive_history_length=int(row["positive_history_length"]),
                    recent_mean_rating=float(row["recent_mean_rating"]),
                    user_sex=row["user_sex"],
                    user_age=int(row["user_age"]),
                    user_occupation=row["user_occupation"],
                    top_genres=tuple(json.loads(row["top_genres_json"])),
                    positive_genre_counter={key: int(value) for key, value in json.loads(row["positive_genre_counter_json"]).items()},
                    observed_item_index=int(row["observed_item_index"]),
                    observed_reward=int(row["observed_reward"]),
                    stage1_action=int(row["stage1_action"]),
                    stage2_action=int(row["stage2_action"]),
                    support_generator_indices=tuple(int(value) for value in json.loads(row["support_generator_indices_json"])),
                    stage1_probs=tuple(float(value) for value in json.loads(row["stage1_probs_json"])),
                    generator_scores=tuple(float(value) for value in json.loads(row["generator_scores_json"])),
                    proxy_values=tuple(float(value) for value in json.loads(row["proxy_values_json"])),
                    candidate_sets=tuple(tuple(int(value) for value in values) for values in json.loads(row["candidate_sets_json"])),
                    stage2_probs=tuple(tuple(float(value) for value in values) for values in json.loads(row["stage2_probs_json"])),
                    seen_movie_count=int(row["seen_movie_count"]),
                )
            )
            if context_cap is not None and len(contexts) >= context_cap:
                break
    return contexts


def build_logged_records(contexts: list[ApplicationContext]) -> list[ApplicationLoggedRecord]:
    return [
        ApplicationLoggedRecord(
            context=context,
            stage1_action=context.stage1_action,
            stage2_action=context.stage2_action,
            reward=float(context.observed_reward),
            proxy=float(context.proxy_values[context.stage1_action]),
        )
        for context in contexts
    ]


class ReconstructedMovieLensEnvironment:
    def __init__(
        self,
        app_config: ApplicationConfig,
        catalog: list[CatalogItem],
        contexts: list[ApplicationContext],
        eval_contexts: list[ApplicationContext] | None = None,
    ) -> None:
        self.app_config = app_config
        self.catalog = catalog
        self.catalog_by_index = {row.item_index: row for row in catalog}
        self.generator_ids = [generator.generator_id for generator in app_config.generators]
        self.contexts = contexts
        self.eval_contexts = eval_contexts if eval_contexts is not None else contexts
        self.config = ApplicationEnvConfig(
            num_generators=len(app_config.generators),
            num_items=len(catalog),
        )
        release_years = [row.release_year for row in catalog if row.release_year is not None]
        self.min_release_year = min(release_years) if release_years else 1900
        self.max_release_year = max(release_years) if release_years else self.min_release_year + 1

    def sample_context(self, rng: Random) -> ApplicationContext:
        return self.eval_contexts[rng.randrange(len(self.eval_contexts))]

    def feasible_set(self, context: ApplicationContext, generator: int) -> list[int]:
        return list(context.candidate_sets[generator])

    def logging_stage1_probs(self, context: ApplicationContext) -> list[float]:
        return list(context.stage1_probs)

    def logging_stage2_probs(self, context: ApplicationContext, generator: int) -> list[float]:
        full = [0.0] * self.config.num_items
        for item, prob in zip(context.candidate_sets[generator], context.stage2_probs[generator]):
            full[item] = prob
        return full

    def _genre_affinity(self, context: ApplicationContext, item: CatalogItem) -> float:
        total = sum(context.positive_genre_counter.values())
        if total <= 0 or not item.genres:
            return 0.0
        return sum(context.positive_genre_counter.get(genre, 0) for genre in item.genres) / (total * len(item.genres))

    def _release_year_norm(self, item: CatalogItem) -> float:
        if item.release_year is None:
            return 0.5
        span = max(self.max_release_year - self.min_release_year, 1)
        return (item.release_year - self.min_release_year) / span

    def _top_genre_match(self, context: ApplicationContext, item: CatalogItem) -> float:
        top_genres = {genre for genre in context.top_genres if genre}
        if not top_genres or not item.genres:
            return 0.0
        return 1.0 if any(genre in top_genres for genre in item.genres) else 0.0

    def _generator_bonus(
        self,
        context: ApplicationContext,
        generator: int,
        item: CatalogItem,
        genre_affinity: float,
        top_genre_match: float,
        release_year_norm: float,
    ) -> float:
        generator_id = self.generator_ids[generator]
        if generator_id == "popularity_head":
            return 0.12 * top_genre_match + 0.08 * release_year_norm
        if generator_id == "genre_match":
            return 0.30 * genre_affinity + 0.10 * top_genre_match
        if generator_id == "collaborative_neighbor":
            positive_history_fraction = context.positive_history_length / max(context.history_length, 1)
            return 0.12 * positive_history_fraction + 0.10 * genre_affinity
        return 0.10 * genre_affinity + 0.08 * (1.0 - release_year_norm)

    def reward_mean(self, context: ApplicationContext, generator: int, item: int) -> float:
        catalog_item = self.catalog_by_index[item]
        genre_affinity = self._genre_affinity(context, catalog_item)
        top_genre_match = self._top_genre_match(context, catalog_item)
        release_year_norm = self._release_year_norm(catalog_item)
        positive_history_fraction = context.positive_history_length / max(context.history_length, 1)
        recent_mean = context.recent_mean_rating / 5.0
        generator_bonus = self._generator_bonus(
            context,
            generator,
            catalog_item,
            genre_affinity,
            top_genre_match,
            release_year_norm,
        )
        latent = (
            -1.05
            + 1.25 * genre_affinity
            + 0.35 * top_genre_match
            + 0.20 * release_year_norm
            + 0.22 * recent_mean
            + 0.15 * positive_history_fraction
            + generator_bonus
        )
        return _sigmoid(latent)

    def reward_features(self, context: ApplicationContext, generator: int, item: int) -> list[float]:
        catalog_item = self.catalog_by_index[item]
        genre_affinity = self._genre_affinity(context, catalog_item)
        top_genre_match = self._top_genre_match(context, catalog_item)
        release_year_norm = self._release_year_norm(catalog_item)
        genre_count_norm = len(catalog_item.genres) / 5.0
        generator_one_hot = [1.0 if index == generator else 0.0 for index in range(self.config.num_generators)]
        positive_history_fraction = context.positive_history_length / max(context.history_length, 1)
        occupation_value = float(context.user_occupation) / 20.0 if context.user_occupation.isdigit() else 0.0
        return [
            1.0,
            context.history_length / 50.0,
            positive_history_fraction,
            context.recent_mean_rating / 5.0,
            1.0 if context.user_sex == "M" else 0.0,
            context.user_age / 56.0,
            occupation_value,
            genre_affinity,
            top_genre_match,
            release_year_norm,
            genre_count_norm,
            *generator_one_hot,
            genre_affinity * release_year_norm,
            genre_affinity * top_genre_match,
        ]

    def proxy_features(self, context: ApplicationContext, generator: int) -> list[float]:
        generator_one_hot = [1.0 if index == generator else 0.0 for index in range(self.config.num_generators)]
        positive_history_fraction = context.positive_history_length / max(context.history_length, 1)
        return [
            1.0,
            context.history_length / 50.0,
            positive_history_fraction,
            context.recent_mean_rating / 5.0,
            context.generator_scores[generator],
            context.proxy_values[generator],
            len(context.support_generator_indices) / max(self.config.num_generators, 1),
            *generator_one_hot,
        ]

    def oracle_stage2(self, context: ApplicationContext, generator: int) -> int:
        feasible = self.feasible_set(context, generator)
        return max(feasible, key=lambda item: (self.reward_mean(context, generator, item), -item))

    def oracle_stage1(self, context: ApplicationContext) -> int:
        scores = []
        for generator in range(self.config.num_generators):
            best_item = self.oracle_stage2(context, generator)
            scores.append(self.reward_mean(context, generator, best_item))
        return max(range(self.config.num_generators), key=lambda generator: (scores[generator], -generator))
