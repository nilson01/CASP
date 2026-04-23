from __future__ import annotations

from .utils import EPS


def zeros(rows: int, cols: int) -> list[list[float]]:
    return [[0.0 for _ in range(cols)] for _ in range(rows)]


def transpose(matrix: list[list[float]]) -> list[list[float]]:
    if not matrix:
        return []
    return [list(column) for column in zip(*matrix)]


def solve_linear_system(matrix: list[list[float]], vector: list[float]) -> list[float]:
    n_rows = len(matrix)
    augmented = [row[:] + [vector[index]] for index, row in enumerate(matrix)]

    for pivot_col in range(n_rows):
        pivot_row = max(range(pivot_col, n_rows), key=lambda row: abs(augmented[row][pivot_col]))
        if abs(augmented[pivot_row][pivot_col]) <= EPS:
            augmented[pivot_row][pivot_col] = EPS
        augmented[pivot_col], augmented[pivot_row] = augmented[pivot_row], augmented[pivot_col]

        pivot_value = augmented[pivot_col][pivot_col]
        for col in range(pivot_col, n_rows + 1):
            augmented[pivot_col][col] /= pivot_value

        for row in range(n_rows):
            if row == pivot_col:
                continue
            factor = augmented[row][pivot_col]
            if abs(factor) <= EPS:
                continue
            for col in range(pivot_col, n_rows + 1):
                augmented[row][col] -= factor * augmented[pivot_col][col]

    return [augmented[row][n_rows] for row in range(n_rows)]


def ridge_regression_fit(
    features: list[list[float]],
    targets: list[float],
    alpha: float,
) -> list[float]:
    n_features = len(features[0])
    gram = zeros(n_features, n_features)
    rhs = [0.0] * n_features

    for row, target in zip(features, targets):
        for left in range(n_features):
            rhs[left] += row[left] * target
            for right in range(n_features):
                gram[left][right] += row[left] * row[right]

    for index in range(n_features):
        gram[index][index] += alpha

    return solve_linear_system(gram, rhs)

