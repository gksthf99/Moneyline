import numpy as np

from src.services.replacement_training import (
    accuracy,
    brier_score,
    chronological_split,
    fit_logistic_model,
)


def test_chronological_split_preserves_order():
    rows = [
        {"game_date": "2026-03-01", "home_team": "A", "away_team": "B"},
        {"game_date": "2026-03-02", "home_team": "C", "away_team": "D"},
        {"game_date": "2026-03-03", "home_team": "E", "away_team": "F"},
        {"game_date": "2026-03-04", "home_team": "G", "away_team": "H"},
    ]
    train, test = chronological_split(rows, train_ratio=0.5)
    assert [row["game_date"] for row in train] == ["2026-03-01", "2026-03-02"]
    assert [row["game_date"] for row in test] == ["2026-03-03", "2026-03-04"]


def test_fit_logistic_model_learns_simple_signal():
    rows = [
        {"feature_a": -2.0, "feature_b": 0.0, "home_won": False},
        {"feature_a": -1.0, "feature_b": 0.0, "home_won": False},
        {"feature_a": 1.0, "feature_b": 0.0, "home_won": True},
        {"feature_a": 2.0, "feature_b": 0.0, "home_won": True},
    ]
    model = fit_logistic_model(rows, ("feature_a", "feature_b"), epochs=800, learning_rate=0.1)
    probs = model.predict_proba(rows)

    assert probs[0] < 0.5
    assert probs[-1] > 0.5


def test_metrics_helpers():
    preds = np.array([0.8, 0.2])
    actuals = np.array([1.0, 0.0])
    assert round(brier_score(preds, actuals), 4) == 0.04
    assert accuracy(preds, actuals) == 1.0
