from __future__ import annotations

import math
from random import Random


EPS = 1e-12


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def sigmoid(value: float) -> float:
    if value >= 0.0:
        exp_term = math.exp(-value)
        return 1.0 / (1.0 + exp_term)
    exp_term = math.exp(value)
    return exp_term / (1.0 + exp_term)


def softmax(scores: list[float]) -> list[float]:
    if not scores:
        return []
    max_score = max(scores)
    shifted = [math.exp(score - max_score) for score in scores]
    total = sum(shifted)
    if total <= EPS:
        return [1.0 / len(scores)] * len(scores)
    return [value / total for value in shifted]


def mix_with_uniform(probs: list[float], mass: float) -> list[float]:
    n_items = len(probs)
    if n_items == 0:
        return []
    uniform = 1.0 / n_items
    return [(1.0 - mass) * prob + mass * uniform for prob in probs]


def renormalize(probs: list[float]) -> list[float]:
    total = sum(probs)
    if total <= EPS:
        return [1.0 / len(probs)] * len(probs)
    return [value / total for value in probs]


def argmax(values: list[float]) -> int:
    best_index = 0
    best_value = values[0]
    for index, value in enumerate(values[1:], start=1):
        if value > best_value:
            best_value = value
            best_index = index
    return best_index


def dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def elementwise(left: list[float], right: list[float]) -> list[float]:
    return [a * b for a, b in zip(left, right)]


def mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[midpoint]
    return 0.5 * (ordered[midpoint - 1] + ordered[midpoint])


def stddev(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    avg = mean(values)
    variance = sum((value - avg) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)


def standard_error(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return stddev(values) / math.sqrt(len(values))


def sample_from_probs(rng: Random, probs: list[float]) -> int:
    threshold = rng.random()
    running = 0.0
    for index, prob in enumerate(probs):
        running += prob
        if threshold <= running:
            return index
    return len(probs) - 1


def one_hot(size: int, index: int) -> list[float]:
    values = [0.0] * size
    values[index] = 1.0
    return values
