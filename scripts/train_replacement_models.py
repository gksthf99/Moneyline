#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from src.services.replacement_training import (
    accuracy,
    brier_score,
    chronological_split,
    fit_logistic_model,
)


NBA_FEATURES = (
    "base_probability",
    "final_probability",
    "strength_spread",
    "home_win_pct",
    "away_win_pct",
    "rest_differential",
    "home_b2b",
    "away_b2b",
    "h2h_adjustment",
)

NHL_FEATURES = (
    "base_probability",
    "final_probability",
    "strength_spread",
    "home_win_pct",
    "away_win_pct",
    "xgf_pct_spread",
    "gf_ga_ratio_spread",
    "rest_differential",
    "home_b2b",
    "away_b2b",
    "home_ice_suppressed",
    "h2h_adjustment",
)


def _as_bool(value: str | None) -> bool:
    return str(value).strip().lower() in {"1", "true", "t", "yes"}


def load_backtest_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as handle:
        for row in csv.DictReader(handle):
            sport = row["sport"].upper()
            rows.append(
                {
                    "sport": sport,
                    "game_date": row.get("game_date", ""),
                    "home_team": row.get("home_team", ""),
                    "away_team": row.get("away_team", ""),
                    "base_probability": float(row.get("base_prob_home") or 0.0),
                    "final_probability": float(row.get("final_prob_home") or 0.0),
                    "strength_spread": float(row.get("home_strength") or 0.0) - float(row.get("away_strength") or 0.0),
                    "home_win_pct": float(row.get("home_win_rate") or 0.0),
                    "away_win_pct": float(row.get("away_win_rate") or 0.0),
                    "xgf_pct_spread": float(row.get("home_xgf_pct") or 0.0) - float(row.get("away_xgf_pct") or 0.0),
                    "gf_ga_ratio_spread": float(row.get("home_gf_ga_ratio") or 0.0) - float(row.get("away_gf_ga_ratio") or 0.0),
                    "rest_differential": float(row.get("home_rest_days") or 0.0) - float(row.get("away_rest_days") or 0.0),
                    "home_b2b": float(_as_bool(row.get("home_b2b"))),
                    "away_b2b": float(_as_bool(row.get("away_b2b"))),
                    "home_ice_suppressed": float(_as_bool(row.get("home_ice_suppressed"))),
                    "h2h_adjustment": float(row.get("h2h_adj") or 0.0),
                    "home_won": _as_bool(row.get("actual_home_win")),
                }
            )
    return rows


def evaluate_sport(rows: list[dict], sport: str, feature_names: tuple[str, ...]) -> dict:
    sport_rows = [row for row in rows if row["sport"] == sport]
    train_rows, test_rows = chronological_split(sport_rows, train_ratio=0.7)
    model = fit_logistic_model(train_rows, feature_names)
    predictions = model.predict_proba(test_rows)
    actuals = np.array([1.0 if row["home_won"] else 0.0 for row in test_rows], dtype=float)
    live_preds = np.array([float(row["final_probability"]) for row in test_rows], dtype=float)
    base_preds = np.array([float(row["base_probability"]) for row in test_rows], dtype=float)

    return {
        "sport": sport,
        "n_train": len(train_rows),
        "n_test": len(test_rows),
        "replacement_brier": brier_score(predictions, actuals),
        "live_brier": brier_score(live_preds, actuals),
        "base_brier": brier_score(base_preds, actuals),
        "replacement_acc": accuracy(predictions, actuals),
        "live_acc": accuracy(live_preds, actuals),
        "base_acc": accuracy(base_preds, actuals),
        "weights": dict(zip(feature_names, model.weights.tolist())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train bootstrap replacement models from backtest CSVs.")
    parser.add_argument("--csv", action="append", required=True, help="Backtest CSV input (repeatable)")
    args = parser.parse_args()

    rows: list[dict] = []
    for csv_path in args.csv:
        rows.extend(load_backtest_rows(Path(csv_path)))

    nba = evaluate_sport(rows, "NBA", NBA_FEATURES)
    nhl = evaluate_sport(rows, "NHL", NHL_FEATURES)

    print("Bootstrap replacement-model evaluation")
    print(f"{'Sport':4s} {'Train':>5s} {'Test':>5s} {'Repl Brier':>11s} {'Live Brier':>11s} {'Base Brier':>11s} {'Repl Acc':>9s} {'Live Acc':>9s}")
    for result in (nba, nhl):
        print(
            f"{result['sport']:4s} {result['n_train']:5d} {result['n_test']:5d} "
            f"{result['replacement_brier']:11.4f} {result['live_brier']:11.4f} {result['base_brier']:11.4f} "
            f"{result['replacement_acc']:9.1%} {result['live_acc']:9.1%}"
        )
        print(f"  top weights: {sorted(result['weights'].items(), key=lambda item: abs(item[1]), reverse=True)[:5]}")


if __name__ == "__main__":
    main()
