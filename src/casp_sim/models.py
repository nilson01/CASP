from __future__ import annotations

from dataclasses import dataclass

from .linalg import ridge_regression_fit
from .utils import clamp, dot


@dataclass
class RidgeRegressor:
    weights: list[float]
    clip_min: float | None = None
    clip_max: float | None = None

    def predict(self, features: list[float]) -> float:
        value = dot(self.weights, features)
        if self.clip_min is not None and self.clip_max is not None:
            return clamp(value, self.clip_min, self.clip_max)
        return value

    @classmethod
    def fit(
        cls,
        features: list[list[float]],
        targets: list[float],
        alpha: float,
        clip_min: float | None = None,
        clip_max: float | None = None,
    ) -> "RidgeRegressor":
        weights = ridge_regression_fit(features, targets, alpha)
        return cls(weights=weights, clip_min=clip_min, clip_max=clip_max)

