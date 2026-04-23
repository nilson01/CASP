from __future__ import annotations

import json
import heapq
import logging
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from random import Random

from .config import ApplicationConfig
from .datasets import extracted_file_manifest_rows
from .reports import write_json, write_rows_csv
from .schema import MovieRecord, RatingRecord, RequestContextRow, UserRecord


def generator_manifest_rows(config: ApplicationConfig) -> list[dict]:
    return [
        {
            "generator_id": generator.generator_id,
            "label": generator.label,
            "description": generator.description,
            "score_formula": generator.score_formula,
            "support_effect": generator.support_effect,
        }
        for generator in config.generators
    ]


def logging_design_rows(config: ApplicationConfig) -> list[dict]:
    return [
        {
            "component": "stage1",
            "construction": "softmax over support-eligible generator scores, followed by an explicit epsilon-uniform exploration mixture inside the supported generator set after mild generator-score shrinkage toward the shared reranker score",
            "temperature": config.stage1_temperature,
            "exploration_epsilon": config.stage1_exploration_epsilon,
            "legacy_min_mass_field": config.min_stage1_mass,
        },
        {
            "component": "stage2",
            "construction": "softmax over reranker scores within the chosen feasible set with minimum mass floor",
            "temperature": config.stage2_temperature,
            "min_mass": config.min_stage2_mass,
        },
    ]


def reconstruction_plan_rows(config: ApplicationConfig) -> list[dict]:
    return [
        {
            "component": "request_context",
            "design_choice": "one rating event per eligible user-time step after warm start",
        },
        {
            "component": "stage1_action",
            "design_choice": "finite generator choice from the locked four-generator library",
        },
        {
            "component": "stage1_rebalance",
            "design_choice": "mild shrinkage of popularity-head and collaborative-neighbor generator scores toward a shared request-level reranker score before top-k candidate construction, followed by an explicit exploratory stage-1 mixture at logging time",
        },
        {
            "component": "generator2_support_redesign",
            "design_choice": f"collaborative-neighbor scoring uses strategy {config.collaborative_neighbor_strategy} to widen generator-2 feasible support through hierarchical demographic backoff before top-k candidate construction",
        },
        {
            "component": "feasible_support",
            "design_choice": f"top-{config.candidate_set_size} candidate set under the chosen generator with already-seen items removed",
        },
        {
            "component": "support_filter",
            "design_choice": f"keep contexts with at least {config.min_supporting_generators} supporting generators, fallback to {config.fallback_min_supporting_generators} only if fewer than {config.fallback_context_floor} contexts remain, then construct the fallback pool with strategy {config.fallback_pool_strategy} and singleton caps {config.fallback_singleton_caps}",
        },
        {
            "component": "stage2_action",
            "design_choice": "observed movie treated as the logged final item inside the reconstructed feasible set",
        },
        {
            "component": "reward",
            "design_choice": f"binary positive label derived from rating >= {config.positive_rating_threshold}",
        },
        {
            "component": "evaluation",
            "design_choice": "hybrid application evaluation with a temporal holdout, DR/burden/stability in main results, and appendix-only reconstructed oracle diagnostics",
        },
    ]


def _softmax(scores: list[float], temperature: float) -> list[float]:
    if not scores:
        return []
    scaled = [score / max(temperature, 1e-9) for score in scores]
    max_score = max(scaled)
    shifted = [math.exp(score - max_score) for score in scaled]
    total = sum(shifted)
    if total <= 1e-12:
        return [1.0 / len(scores)] * len(scores)
    return [value / total for value in shifted]


def _mix_with_uniform(probs: list[float], mass: float) -> list[float]:
    if not probs:
        return []
    uniform = 1.0 / len(probs)
    return [(1.0 - mass) * prob + mass * uniform for prob in probs]


def _sample_from_probs(rng: Random, probs: list[float]) -> int:
    threshold = rng.random()
    running = 0.0
    for index, prob in enumerate(probs):
        running += prob
        if threshold <= running:
            return index
    return len(probs) - 1


def _smoothed_rate(positive_count: int, total_count: int, prior_rate: float, alpha: float) -> float:
    return (positive_count + alpha * prior_rate) / (total_count + alpha) if total_count > 0 else prior_rate


def _reservoir_append_or_replace(
    reservoir_rows: list[dict],
    seen_count: int,
    cap: int,
    rng: Random,
    row: dict,
) -> None:
    if len(reservoir_rows) < cap:
        reservoir_rows.append(row)
        return
    replace_index = rng.randrange(seen_count)
    if replace_index < cap:
        reservoir_rows[replace_index] = row


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _genre_affinity(
    movie: MovieRecord,
    positive_genre_counter: Counter[str],
    total_positive_genres: int,
) -> float:
    if not movie.genres:
        return 0.0
    if total_positive_genres <= 0:
        return 0.0
    return _mean([positive_genre_counter.get(genre, 0) / total_positive_genres for genre in movie.genres])


def _generator_score(generator_id: str, row: dict[str, float]) -> float:
    if generator_id == "popularity_head":
        return (
            0.70 * row["popularity_norm"]
            + 0.20 * row["mean_rating_norm"]
            + 0.10 * row["genre_affinity"]
        )
    if generator_id == "genre_match":
        return (
            0.75 * row["genre_affinity"]
            + 0.15 * row["mean_rating_norm"]
            + 0.10 * row["novelty_score"]
        )
    if generator_id == "collaborative_neighbor":
        return (
            0.55 * row["demographic_neighbor"]
            + 0.20 * row["genre_affinity"]
            + 0.15 * row["mean_rating_norm"]
            + 0.10 * row["popularity_norm"]
        )
    return (
        0.55 * row["genre_affinity"]
        + 0.35 * row["novelty_score"]
        + 0.10 * row["mean_rating_norm"]
    )


def _rebalanced_generator_score(
    generator_id: str,
    base_score: float,
    reranker_score: float,
) -> float:
    base_weight = {
        "popularity_head": 0.75,
        "genre_match": 0.90,
        "collaborative_neighbor": 0.85,
        "long_tail_explorer": 0.90,
    }[generator_id]
    return base_weight * base_score + (1.0 - base_weight) * reranker_score


def catalog_rows(
    config: ApplicationConfig,
    movies: list[MovieRecord],
    ratings: list[RatingRecord],
) -> list[dict]:
    rating_counter: Counter[int] = Counter()
    positive_counter: Counter[int] = Counter()
    rating_sum: defaultdict[int, float] = defaultdict(float)
    for rating in ratings:
        rating_counter[rating.movie_id] += 1
        rating_sum[rating.movie_id] += rating.rating
        if rating.rating >= config.positive_rating_threshold:
            positive_counter[rating.movie_id] += 1

    max_count = max(rating_counter.values()) if rating_counter else 1
    rows = []
    for item_index, movie in enumerate(sorted(movies, key=lambda row: row.movie_id)):
        count = rating_counter[movie.movie_id]
        mean_rating = rating_sum[movie.movie_id] / count if count else 0.0
        popularity_norm = count / max_count if max_count else 0.0
        positive_rate = positive_counter[movie.movie_id] / count if count else 0.0
        rows.append(
            {
                "item_index": item_index,
                "movie_id": movie.movie_id,
                "title": movie.title,
                "release_year": movie.release_year if movie.release_year is not None else "",
                "genres": "|".join(movie.genres),
                "rating_count": count,
                "mean_rating": round(mean_rating, 6),
                "positive_rate": round(positive_rate, 6),
                "popularity_norm": round(popularity_norm, 6),
                "novelty_score": round(1.0 - popularity_norm, 6),
            }
        )
    return rows


def request_context_rows(
    config: ApplicationConfig,
    users: list[UserRecord],
    movies: list[MovieRecord],
    ratings: list[RatingRecord],
) -> list[dict]:
    prepared = _prepare_request_rows(config, users, movies, ratings)
    return prepared["final_rows"]


def _prepare_request_rows(
    config: ApplicationConfig,
    users: list[UserRecord],
    movies: list[MovieRecord],
    ratings: list[RatingRecord],
    logger: logging.Logger | None = None,
) -> dict:
    start_time = time.monotonic()
    last_log_time = start_time
    users_by_id = {user.user_id: user for user in users}
    movies_sorted = sorted(movies, key=lambda row: row.movie_id)
    movies_by_id = {movie.movie_id: movie for movie in movies_sorted}
    item_index_by_movie_id = {movie.movie_id: index for index, movie in enumerate(movies_sorted)}
    ratings_sorted = sorted(ratings, key=lambda row: (row.timestamp, row.user_id, row.movie_id))
    user_rating_counts = Counter(rating.user_id for rating in ratings_sorted)
    eligible_user_count = sum(
        1
        for count in user_rating_counts.values()
        if count > config.min_history_length
    )

    rating_counter: Counter[int] = Counter()
    positive_counter: Counter[int] = Counter()
    rating_sum: defaultdict[int, float] = defaultdict(float)
    bucket_positive_counter: defaultdict[tuple[str, int], Counter[int]] = defaultdict(Counter)
    bucket_rating_counter: defaultdict[tuple[str, int], Counter[int]] = defaultdict(Counter)
    sex_positive_counter: defaultdict[str, Counter[int]] = defaultdict(Counter)
    sex_rating_counter: defaultdict[str, Counter[int]] = defaultdict(Counter)
    age_positive_counter: defaultdict[int, Counter[int]] = defaultdict(Counter)
    age_rating_counter: defaultdict[int, Counter[int]] = defaultdict(Counter)
    max_rating_count = 1

    user_histories: defaultdict[int, list[RatingRecord]] = defaultdict(list)
    user_seen_movie_ids: defaultdict[int, set[int]] = defaultdict(set)
    user_positive_history_lengths: defaultdict[int, int] = defaultdict(int)
    user_positive_genre_masses: defaultdict[int, int] = defaultdict(int)
    user_positive_genre_counters: defaultdict[int, Counter[str]] = defaultdict(Counter)

    strict_rows: list[dict] = []
    fallback_only_rows: list[dict] = []
    fallback_reservoir_rng = Random(config.seed + 424242)
    fallback_singleton_rows_by_generator: defaultdict[int, list[dict]] = defaultdict(list)
    fallback_singleton_seen_counter: Counter[int] = Counter()
    fallback_singleton_rng_by_generator: dict[int, Random] = {}
    strict_eligible_count = 0
    fallback_eligible_count = 0
    fallback_only_eligible_count = 0
    fallback_singleton_eligible_counter: Counter[int] = Counter()
    request_id = 0
    examined_events = 0

    if logger is not None:
        logger.info(
            "Starting reconstruction over %s ratings from %s eligible users with context_limit=%s and candidate_set_size=%s",
            len(ratings_sorted),
            eligible_user_count,
            config.context_limit,
            config.candidate_set_size,
        )

    for rating_index, current in enumerate(ratings_sorted, start=1):
        user_id = current.user_id
        if user_rating_counts[user_id] <= config.min_history_length:
            bucket = (users_by_id[user_id].sex, users_by_id[user_id].age)
            user_histories[user_id].append(current)
            user_seen_movie_ids[user_id].add(current.movie_id)
            if current.rating >= config.positive_rating_threshold:
                user_positive_history_lengths[user_id] += 1
                for genre in movies_by_id[current.movie_id].genres:
                    user_positive_genre_counters[user_id][genre] += 1
                    user_positive_genre_masses[user_id] += 1
            rating_counter[current.movie_id] += 1
            rating_sum[current.movie_id] += current.rating
            bucket_rating_counter[bucket][current.movie_id] += 1
            sex_rating_counter[users_by_id[user_id].sex][current.movie_id] += 1
            age_rating_counter[users_by_id[user_id].age][current.movie_id] += 1
            if current.rating >= config.positive_rating_threshold:
                positive_counter[current.movie_id] += 1
                bucket_positive_counter[bucket][current.movie_id] += 1
                sex_positive_counter[users_by_id[user_id].sex][current.movie_id] += 1
                age_positive_counter[users_by_id[user_id].age][current.movie_id] += 1
            max_rating_count = max(max_rating_count, rating_counter[current.movie_id])
            continue

        user = users_by_id[user_id]
        bucket = (user.sex, user.age)
        history = user_histories[user_id]
        seen_movie_ids = user_seen_movie_ids[user_id]
        positive_history_length = user_positive_history_lengths[user_id]
        positive_genre_mass = user_positive_genre_masses[user_id]
        positive_genre_counter = user_positive_genre_counters[user_id]

        if len(history) >= config.min_history_length and current.movie_id not in seen_movie_ids:
            observed_movie = movies_by_id[current.movie_id]
            observed_item_index = item_index_by_movie_id[current.movie_id]
            examined_events += 1

            top_genres = [genre for genre, _ in positive_genre_counter.most_common(3)]
            while len(top_genres) < 3:
                top_genres.append("")

            available_count = len(movies_sorted) - len(seen_movie_ids)
            if available_count >= config.candidate_set_size:
                generator_heaps: list[list[tuple[float, int, int, float]]] = [
                    [] for _ in config.generators
                ]
                total_positive_genres = positive_genre_mass
                for movie in movies_sorted:
                    if movie.movie_id in seen_movie_ids:
                        continue
                    count = rating_counter[movie.movie_id]
                    mean_rating_norm = (rating_sum[movie.movie_id] / count) / 5.0 if count else 0.0
                    positive_rate = positive_counter[movie.movie_id] / count if count else 0.0
                    popularity_norm = count / max_rating_count if max_rating_count else 0.0
                    novelty_score = 1.0 - popularity_norm
                    genre_affinity = _genre_affinity(movie, positive_genre_counter, total_positive_genres)
                    bucket_ratings = bucket_rating_counter[bucket][movie.movie_id]
                    bucket_positive = bucket_positive_counter[bucket][movie.movie_id]
                    sex_ratings = sex_rating_counter[user.sex][movie.movie_id]
                    sex_positive = sex_positive_counter[user.sex][movie.movie_id]
                    age_ratings = age_rating_counter[user.age][movie.movie_id]
                    age_positive = age_positive_counter[user.age][movie.movie_id]
                    exact_neighbor = _smoothed_rate(bucket_positive, bucket_ratings, positive_rate, alpha=20.0)
                    age_neighbor = _smoothed_rate(age_positive, age_ratings, positive_rate, alpha=35.0)
                    sex_neighbor = _smoothed_rate(sex_positive, sex_ratings, positive_rate, alpha=35.0)
                    demographic_neighbor = (
                        0.55 * exact_neighbor
                        + 0.25 * age_neighbor
                        + 0.10 * sex_neighbor
                        + 0.10 * positive_rate
                    )
                    item_index = item_index_by_movie_id[movie.movie_id]
                    feature_row = {
                        "mean_rating_norm": mean_rating_norm,
                        "popularity_norm": popularity_norm,
                        "novelty_score": novelty_score,
                        "genre_affinity": genre_affinity,
                        "demographic_neighbor": demographic_neighbor,
                    }
                    reranker_score = (
                        0.45 * genre_affinity
                        + 0.25 * mean_rating_norm
                        + 0.15 * demographic_neighbor
                        + 0.15 * novelty_score
                    )
                    for generator_index, generator in enumerate(config.generators):
                        generator_score = _rebalanced_generator_score(
                            generator.generator_id,
                            _generator_score(generator.generator_id, feature_row),
                            reranker_score,
                        )
                        entry = (generator_score, -item_index, item_index, reranker_score)
                        heap = generator_heaps[generator_index]
                        if len(heap) < config.candidate_set_size:
                            heapq.heappush(heap, entry)
                        elif entry[:2] > heap[0][:2]:
                            heapq.heapreplace(heap, entry)

                generator_candidate_sets: list[list[int]] = []
                generator_stage2_probs: list[list[float]] = []
                generator_scores: list[float] = []
                proxy_values: list[float] = []

                for generator_index, _generator in enumerate(config.generators):
                    top_entries = sorted(
                        generator_heaps[generator_index],
                        key=lambda entry: (-entry[0], -entry[1]),
                    )
                    candidate_items = [entry[2] for entry in top_entries]
                    candidate_scores = [entry[3] for entry in top_entries]
                    stage2_probs = _mix_with_uniform(
                        _softmax(candidate_scores, config.stage2_temperature),
                        config.min_stage2_mass,
                    )
                    proxy_value = _mean(candidate_scores)
                    generator_score_summary = _mean(
                        [entry[0] for entry in top_entries[: min(5, len(top_entries))]]
                    )
                    generator_candidate_sets.append(candidate_items)
                    generator_stage2_probs.append(stage2_probs)
                    generator_scores.append(generator_score_summary)
                    proxy_values.append(proxy_value)

                support_generators = [
                    generator_index
                    for generator_index, candidate_items in enumerate(generator_candidate_sets)
                    if observed_item_index in candidate_items
                ]
                support_count = len(support_generators)

                if support_count > 0:
                    stage1_probs = [0.0] * len(config.generators)
                    supported_scores = [generator_scores[index] for index in support_generators]
                    supported_probs = _mix_with_uniform(
                        _softmax(supported_scores, config.stage1_temperature),
                        config.stage1_exploration_epsilon,
                    )
                    for generator_index, prob in zip(support_generators, supported_probs):
                        stage1_probs[generator_index] = prob

                    rng = Random(config.seed + 997 * request_id)
                    logged_support_index = _sample_from_probs(rng, supported_probs)
                    stage1_action = support_generators[logged_support_index]

                    request = RequestContextRow(
                        request_id=request_id,
                        user_id=user_id,
                        timestamp=current.timestamp,
                        history_length=len(history),
                        positive_history_length=positive_history_length,
                        recent_mean_rating=_mean([row.rating for row in history[-5:]]),
                        top_genre_1=top_genres[0],
                        top_genre_2=top_genres[1],
                        observed_movie_id=current.movie_id,
                        observed_rating=current.rating,
                        observed_reward=int(current.rating >= config.positive_rating_threshold),
                    )
                    row = {
                        **request.to_dict(),
                        "user_sex": user.sex,
                        "user_age": user.age,
                        "user_occupation": user.occupation,
                        "observed_movie_title": observed_movie.title,
                        "observed_movie_genres": "|".join(observed_movie.genres),
                        "observed_item_index": observed_item_index,
                        "support_generator_count": support_count,
                        "support_generator_indices_json": json.dumps(support_generators),
                        "stage1_action": stage1_action,
                        "stage2_action": observed_item_index,
                        "stage1_probs_json": json.dumps(stage1_probs),
                        "generator_scores_json": json.dumps(generator_scores),
                        "proxy_values_json": json.dumps(proxy_values),
                        "candidate_sets_json": json.dumps(generator_candidate_sets),
                        "stage2_probs_json": json.dumps(generator_stage2_probs),
                        "seen_movie_count": len(seen_movie_ids),
                        "top_genres_json": json.dumps(top_genres),
                        "positive_genre_counter_json": json.dumps(dict(positive_genre_counter)),
                    }
                    fallback_eligible_count += 1
                    if support_count >= config.min_supporting_generators:
                        strict_eligible_count += 1
                        if len(strict_rows) < config.context_limit:
                            strict_rows.append(row)
                    else:
                        fallback_only_eligible_count += 1
                        fallback_singleton_eligible_counter[support_generators[0]] += 1
                        if config.fallback_pool_strategy in {
                            "singleton_diversity_preserving",
                            "singleton_diversity_preserving_cap_g1_v1",
                        }:
                            singleton_generator_index = support_generators[0]
                            fallback_singleton_seen_counter[singleton_generator_index] += 1
                            generator_rng = fallback_singleton_rng_by_generator.setdefault(
                                singleton_generator_index,
                                Random(config.seed + 424242 + 9973 * singleton_generator_index),
                            )
                            _reservoir_append_or_replace(
                                reservoir_rows=fallback_singleton_rows_by_generator[singleton_generator_index],
                                seen_count=fallback_singleton_seen_counter[singleton_generator_index],
                                cap=config.context_limit,
                                rng=generator_rng,
                                row=row,
                            )
                        else:
                            _reservoir_append_or_replace(
                                reservoir_rows=fallback_only_rows,
                                seen_count=fallback_only_eligible_count,
                                cap=config.context_limit,
                                rng=fallback_reservoir_rng,
                                row=row,
                            )
                    request_id += 1

        history.append(current)
        seen_movie_ids.add(current.movie_id)
        if current.rating >= config.positive_rating_threshold:
            user_positive_history_lengths[user_id] += 1
            for genre in movies_by_id[current.movie_id].genres:
                positive_genre_counter[genre] += 1
                user_positive_genre_masses[user_id] += 1
        rating_counter[current.movie_id] += 1
        rating_sum[current.movie_id] += current.rating
        bucket_rating_counter[bucket][current.movie_id] += 1
        sex_rating_counter[user.sex][current.movie_id] += 1
        age_rating_counter[user.age][current.movie_id] += 1
        if current.rating >= config.positive_rating_threshold:
            positive_counter[current.movie_id] += 1
            bucket_positive_counter[bucket][current.movie_id] += 1
            sex_positive_counter[user.sex][current.movie_id] += 1
            age_positive_counter[user.age][current.movie_id] += 1
        max_rating_count = max(max_rating_count, rating_counter[current.movie_id])

        if logger is not None:
            now = time.monotonic()
            if now - last_log_time >= 20.0:
                elapsed = now - start_time
                progress_pct = 100.0 * rating_index / max(len(ratings_sorted), 1)
                rows_per_sec = rating_index / max(elapsed, 1e-9)
                remaining_sec = max(0.0, (len(ratings_sorted) - rating_index) / max(rows_per_sec, 1e-9))
                logger.info(
                    "Reconstruction heartbeat: ratings=%s/%s progress_pct=%.2f examined_events=%s kept_strict=%s kept_fallback=%s eligible_rows=%s elapsed_sec=%.1f eta_min=%.1f",
                    rating_index,
                    len(ratings_sorted),
                    progress_pct,
                    examined_events,
                    strict_eligible_count,
                    fallback_eligible_count,
                    fallback_eligible_count,
                    elapsed,
                    remaining_sec / 60.0,
                )
                last_log_time = now

    if strict_eligible_count >= config.fallback_context_floor:
        final_rows = strict_rows[: config.context_limit]
        final_support_threshold = config.min_supporting_generators
        fallback_used = False
        support_pool_mode = "strict_only"
        final_strict_count = len(final_rows)
        final_fallback_only_count = 0
    else:
        strict_part = strict_rows[: config.context_limit]
        remaining_slots = max(0, config.context_limit - len(strict_part))
        if config.fallback_pool_strategy == "singleton_diversity_preserving":
            priority_generators = [
                generator_index
                for generator_index in range(len(config.generators))
                if generator_index != config.fallback_head_generator_index
            ]
            fallback_part: list[dict] = []
            for generator_index in priority_generators:
                fallback_part.extend(fallback_singleton_rows_by_generator.get(generator_index, []))
            if len(fallback_part) < remaining_slots:
                head_rows = fallback_singleton_rows_by_generator.get(config.fallback_head_generator_index, [])
                fallback_part.extend(head_rows[: max(0, remaining_slots - len(fallback_part))])
            fallback_part = sorted(
                fallback_part[:remaining_slots],
                key=lambda row: (row["timestamp"], row["request_id"]),
            )
        elif config.fallback_pool_strategy == "singleton_diversity_preserving_cap_g1_v1":
            priority_generators = [
                generator_index
                for generator_index in range(len(config.generators))
                if generator_index != config.fallback_head_generator_index
            ]
            fallback_part = []
            for generator_index in priority_generators:
                cap = config.fallback_singleton_caps[generator_index]
                fallback_part.extend(fallback_singleton_rows_by_generator.get(generator_index, [])[:cap])
            if len(fallback_part) < remaining_slots:
                head_cap = config.fallback_singleton_caps[config.fallback_head_generator_index]
                head_rows = fallback_singleton_rows_by_generator.get(config.fallback_head_generator_index, [])
                fallback_part.extend(head_rows[: min(head_cap, max(0, remaining_slots - len(fallback_part)))])
            fallback_part = sorted(
                fallback_part[:remaining_slots],
                key=lambda row: (row["timestamp"], row["request_id"]),
            )
        else:
            fallback_part = fallback_only_rows[:remaining_slots]
        final_rows = sorted(
            strict_part + fallback_part,
            key=lambda row: (row["timestamp"], row["request_id"]),
        )
        final_support_threshold = config.fallback_min_supporting_generators
        fallback_used = True
        if config.fallback_pool_strategy == "singleton_diversity_preserving":
            support_pool_mode = "blended_strict_plus_diverse_singleton_fallback"
        elif config.fallback_pool_strategy == "singleton_diversity_preserving_cap_g1_v1":
            support_pool_mode = "blended_strict_plus_capped_diverse_singleton_fallback"
        else:
            support_pool_mode = "blended_strict_plus_fallback"
        final_strict_count = len(strict_part)
        final_fallback_only_count = len(fallback_part)

    fallback_singleton_final_counter: Counter[int] = Counter()
    for row in final_rows:
        support_generators = json.loads(row["support_generator_indices_json"])
        if len(support_generators) == 1:
            fallback_singleton_final_counter[int(support_generators[0])] += 1

    support_rows = [
        {"field": "all_eligible_contexts", "value": fallback_eligible_count},
        {"field": "strict_contexts", "value": strict_eligible_count},
        {"field": "fallback_contexts", "value": fallback_eligible_count},
        {"field": "fallback_only_contexts", "value": fallback_only_eligible_count},
        {"field": "final_contexts", "value": len(final_rows)},
        {"field": "strict_support_threshold", "value": config.min_supporting_generators},
        {"field": "fallback_support_threshold", "value": config.fallback_min_supporting_generators},
        {"field": "fallback_context_floor", "value": config.fallback_context_floor},
        {"field": "fallback_pool_strategy", "value": config.fallback_pool_strategy},
        {"field": "fallback_head_generator_index", "value": config.fallback_head_generator_index},
        {"field": "fallback_singleton_caps", "value": "|".join(str(value) for value in config.fallback_singleton_caps)},
        {"field": "collaborative_neighbor_strategy", "value": config.collaborative_neighbor_strategy},
        {"field": "final_support_threshold", "value": final_support_threshold},
        {"field": "fallback_used", "value": fallback_used},
        {"field": "support_pool_mode", "value": support_pool_mode},
        {"field": "final_strict_contexts", "value": final_strict_count},
        {"field": "final_fallback_only_contexts", "value": final_fallback_only_count},
        {"field": "context_limit", "value": config.context_limit},
        {"field": "fallback_pool_hit_cap", "value": fallback_eligible_count >= config.context_limit},
        {"field": "fallback_pool_reservoir_sampling", "value": True},
        {"field": "full_temporal_scan", "value": True},
        {"field": "temporal_construction", "value": True},
        {"field": "eligible_user_count", "value": eligible_user_count},
        {"field": "examined_events", "value": examined_events},
        {
            "field": "fallback_non_head_singleton_eligible_count",
            "value": sum(
                fallback_singleton_eligible_counter.get(generator_index, 0)
                for generator_index in range(len(config.generators))
                if generator_index != config.fallback_head_generator_index
            ),
        },
        {
            "field": "final_non_head_singleton_count",
            "value": sum(
                fallback_singleton_final_counter.get(generator_index, 0)
                for generator_index in range(len(config.generators))
                if generator_index != config.fallback_head_generator_index
            ),
        },
        {"field": "fallback_singleton_generator_0_eligible_count", "value": fallback_singleton_eligible_counter.get(0, 0)},
        {"field": "fallback_singleton_generator_1_eligible_count", "value": fallback_singleton_eligible_counter.get(1, 0)},
        {"field": "fallback_singleton_generator_2_eligible_count", "value": fallback_singleton_eligible_counter.get(2, 0)},
        {"field": "fallback_singleton_generator_3_eligible_count", "value": fallback_singleton_eligible_counter.get(3, 0)},
        {"field": "final_singleton_generator_0_count", "value": fallback_singleton_final_counter.get(0, 0)},
        {"field": "final_singleton_generator_1_count", "value": fallback_singleton_final_counter.get(1, 0)},
        {"field": "final_singleton_generator_2_count", "value": fallback_singleton_final_counter.get(2, 0)},
        {"field": "final_singleton_generator_3_count", "value": fallback_singleton_final_counter.get(3, 0)},
    ]

    if logger is not None:
        logger.info(
            "Finished reconstruction: examined_events=%s strict_rows=%s fallback_rows=%s final_rows=%s fallback_used=%s elapsed_sec=%.1f",
            examined_events,
            strict_eligible_count,
            fallback_eligible_count,
            len(final_rows),
            fallback_used,
            time.monotonic() - start_time,
        )

    return {
        "catalog_rows": catalog_rows(config, movies, ratings),
        "strict_rows": strict_rows,
        "fallback_rows": fallback_only_rows,
        "final_rows": final_rows,
        "all_rows": final_rows,
        "support_rows": support_rows,
        "fallback_used": fallback_used,
        "final_support_threshold": final_support_threshold,
        "support_pool_mode": support_pool_mode,
    }


def prepare_reconstructed_dataset(
    config: ApplicationConfig,
    users: list[UserRecord],
    movies: list[MovieRecord],
    ratings: list[RatingRecord],
    processed_root: Path,
    logger: logging.Logger | None = None,
) -> dict:
    processed_root.mkdir(parents=True, exist_ok=True)
    prepared = _prepare_request_rows(config, users, movies, ratings, logger=logger)

    if logger is not None:
        logger.info("Writing prepared dataset artifacts to %s", processed_root)

    write_rows_csv(processed_root / "catalog.csv", prepared["catalog_rows"])
    write_rows_csv(processed_root / "request_contexts.csv", prepared["final_rows"])
    write_rows_csv(processed_root / "support_filter_diagnostics.csv", prepared["support_rows"])
    write_rows_csv(processed_root / "extracted_file_manifest.csv", extracted_file_manifest_rows(config))

    manifest = {
        "dataset": config.name,
        "prepared_context_count": len(prepared["final_rows"]),
        "fallback_used": prepared["fallback_used"],
        "final_support_threshold": prepared["final_support_threshold"],
        "support_pool_mode": prepared["support_pool_mode"],
        "context_limit": config.context_limit,
        "candidate_set_size": config.candidate_set_size,
        "min_history_length": config.min_history_length,
        "positive_rating_threshold": config.positive_rating_threshold,
    }
    write_json(processed_root / "preparation_manifest.json", manifest)
    write_rows_csv(processed_root / "preparation_manifest.csv", [manifest])
    if logger is not None:
        logger.info(
            "Preparation manifest written with prepared_context_count=%s final_support_threshold=%s fallback_used=%s",
            manifest["prepared_context_count"],
            manifest["final_support_threshold"],
            manifest["fallback_used"],
        )
    return manifest
