from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LogisticModel:
    weights: np.ndarray
    bias: float
    means: np.ndarray
    scales: np.ndarray
    feature_names: tuple[str, ...]

    def predict_proba(self, rows: list[dict]) -> np.ndarray:
        matrix = np.array(
            [[float(row.get(name, 0.0) or 0.0) for name in self.feature_names] for row in rows],
            dtype=float,
        )
        standardized = (matrix - self.means) / self.scales
        logits = standardized @ self.weights + self.bias
        return 1.0 / (1.0 + np.exp(-np.clip(logits, -35, 35)))


def fit_logistic_model(
    rows: list[dict],
    feature_names: tuple[str, ...],
    target_name: str = "home_won",
    *,
    learning_rate: float = 0.05,
    l2_penalty: float = 0.01,
    epochs: int = 1200,
) -> LogisticModel:
    matrix = np.array(
        [[float(row.get(name, 0.0) or 0.0) for name in feature_names] for row in rows],
        dtype=float,
    )
    labels = np.array([1.0 if row[target_name] else 0.0 for row in rows], dtype=float)

    means = matrix.mean(axis=0)
    scales = matrix.std(axis=0)
    scales[scales == 0] = 1.0
    x = (matrix - means) / scales

    weights = np.zeros(x.shape[1], dtype=float)
    bias = 0.0

    for _ in range(epochs):
        logits = x @ weights + bias
        preds = 1.0 / (1.0 + np.exp(-np.clip(logits, -35, 35)))
        error = preds - labels
        grad_w = (x.T @ error) / len(x) + (l2_penalty * weights)
        grad_b = error.mean()
        weights -= learning_rate * grad_w
        bias -= learning_rate * grad_b

    return LogisticModel(
        weights=weights,
        bias=bias,
        means=means,
        scales=scales,
        feature_names=feature_names,
    )


def brier_score(predictions: np.ndarray, actuals: np.ndarray) -> float:
    return float(np.mean((predictions - actuals) ** 2))


def accuracy(predictions: np.ndarray, actuals: np.ndarray) -> float:
    return float(np.mean((predictions >= 0.5) == (actuals >= 0.5)))


def chronological_split(rows: list[dict], train_ratio: float = 0.7) -> tuple[list[dict], list[dict]]:
    ordered = sorted(rows, key=lambda row: (str(row.get("game_date", "")), str(row.get("home_team", "")), str(row.get("away_team", ""))))
    cutoff = max(1, int(len(ordered) * train_ratio))
    cutoff = min(cutoff, len(ordered) - 1) if len(ordered) > 1 else len(ordered)
    return ordered[:cutoff], ordered[cutoff:]
