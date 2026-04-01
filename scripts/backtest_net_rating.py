#!/usr/bin/env python3
"""
Backtest grid search for NBA net rating model parameters.

Uses existing backtest CSV + current NBA.com net ratings to find optimal
K (logistic scaling) and shrinkage values. Compares against the old
win%-Elo model's Brier score.

Caveat: Uses end-of-season net ratings for the full season (mild look-ahead).
This is acceptable for parameter tuning — the live model uses current-date
ratings which will be slightly less accurate early in the season.

Usage:
    python scripts/backtest_net_rating.py
    python scripts/backtest_net_rating.py --csv backtest_results/nba_nhl_backtest_20260323_015239.csv
"""

import argparse
import csv
import math
import os
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from dotenv import load_dotenv
load_dotenv(PROJECT_DIR / ".env")

from src.data.nba_stats import get_team_ratings, get_team_ratings_recent, get_team_ratings_clutch


# ESPN abbreviation → NBA.com display name mapping
ABBREV_TO_NAME = {
    "ATL": "Atlanta Hawks", "BOS": "Boston Celtics", "BKN": "Brooklyn Nets",
    "CHA": "Charlotte Hornets", "CHI": "Chicago Bulls", "CLE": "Cleveland Cavaliers",
    "DAL": "Dallas Mavericks", "DEN": "Denver Nuggets", "DET": "Detroit Pistons",
    "GS": "Golden State Warriors", "HOU": "Houston Rockets", "IND": "Indiana Pacers",
    "LAC": "LA Clippers", "LAL": "Los Angeles Lakers", "MEM": "Memphis Grizzlies",
    "MIA": "Miami Heat", "MIL": "Milwaukee Bucks", "MIN": "Minnesota Timberwolves",
    "NO": "New Orleans Pelicans", "NY": "New York Knicks", "NYK": "New York Knicks",
    "OKC": "Oklahoma City Thunder", "ORL": "Orlando Magic", "PHI": "Philadelphia 76ers",
    "PHX": "Phoenix Suns", "POR": "Portland Trail Blazers", "SAC": "Sacramento Kings",
    "SA": "San Antonio Spurs", "TOR": "Toronto Raptors", "UTAH": "Utah Jazz",
    "UTA": "Utah Jazz", "WSH": "Washington Wizards", "WAS": "Washington Wizards",
}


def brier(predicted: float, actual: bool) -> float:
    return (predicted - int(actual)) ** 2


def net_rtg_probability(
    home_net_rtg: float,
    away_net_rtg: float,
    home_l10_net: float | None,
    away_l10_net: float | None,
    home_clutch_net: float | None,
    away_clutch_net: float | None,
    k: float,
    shrinkage: float,
    hca: float = 2.5,
    sos_weight: float = 0.60,
    form_weight: float = 0.40,
    overall_weight: float = 0.85,
    clutch_weight: float = 0.15,
) -> float:
    """Compute win probability from net ratings with given parameters."""
    # Garbage time mitigation
    h_net = overall_weight * home_net_rtg + clutch_weight * (home_clutch_net or home_net_rtg)
    a_net = overall_weight * away_net_rtg + clutch_weight * (away_clutch_net or away_net_rtg)

    # Blend season + form
    h_form = home_l10_net if home_l10_net is not None else h_net
    a_form = away_l10_net if away_l10_net is not None else a_net

    h_strength = sos_weight * h_net + form_weight * h_form
    a_strength = sos_weight * a_net + form_weight * a_form

    # Logistic with shrinkage
    diff = (h_strength - a_strength + hca) * shrinkage
    return 1.0 / (1.0 + 10 ** (-diff / k))


def run(csv_path: str):
    # Load ratings
    print("Fetching NBA.com ratings...")
    season = get_team_ratings()
    l10 = get_team_ratings_recent(10)
    clutch = get_team_ratings_clutch()

    if not season:
        print("ERROR: Failed to fetch NBA ratings")
        sys.exit(1)
    print(f"  Season: {len(season)} teams, L10: {len(l10)}, Clutch: {len(clutch)}")

    # Load backtest CSV
    print(f"\nLoading backtest: {csv_path}")
    nba_games = []
    old_brier_sum = 0.0

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["sport"] != "nba":
                continue
            if not row.get("actual_home_win") or row["actual_home_win"] == "":
                continue

            home_abbr = row["home_team"]
            away_abbr = row["away_team"]
            home_name = ABBREV_TO_NAME.get(home_abbr)
            away_name = ABBREV_TO_NAME.get(away_abbr)

            if not home_name or not away_name:
                continue
            if home_name not in season or away_name not in season:
                continue

            actual = row["actual_home_win"].lower() in ("true", "1", "yes")
            old_brier = float(row.get("brier_score", 0))

            nba_games.append({
                "home_name": home_name,
                "away_name": away_name,
                "actual": actual,
                "old_brier": old_brier,
                "old_prob": float(row.get("final_prob_home", 0.5)),
            })
            old_brier_sum += old_brier

    n = len(nba_games)
    old_mean = old_brier_sum / n if n else 0
    print(f"  NBA games with outcomes: {n}")
    print(f"  Old model (win% Elo) Brier: {old_mean:.4f}")

    # Grid search
    print("\nGrid search: K × Shrinkage")
    print(f"{'K':>6s} {'Shrink':>8s} {'Brier':>8s} {'Δ vs old':>10s} {'Better':>8s}")
    print("-" * 44)

    best_brier = 999
    best_k = 0
    best_shrink = 0

    for k in [10.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 20.0, 22.0, 25.0]:
        for shrink in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 1.00]:
            total_brier = 0.0
            for g in nba_games:
                h_s = season[g["home_name"]]
                a_s = season[g["away_name"]]
                h_l = l10.get(g["home_name"])
                a_l = l10.get(g["away_name"])
                h_c = clutch.get(g["home_name"])
                a_c = clutch.get(g["away_name"])

                prob = net_rtg_probability(
                    h_s["net_rtg"], a_s["net_rtg"],
                    h_l["net_rtg"] if h_l else None,
                    a_l["net_rtg"] if a_l else None,
                    h_c["net_rtg"] if h_c else None,
                    a_c["net_rtg"] if a_c else None,
                    k=k, shrinkage=shrink,
                )
                total_brier += brier(prob, g["actual"])

            mean = total_brier / n
            delta = mean - old_mean
            is_better = "  YES" if mean < old_mean else ""

            if mean < best_brier:
                best_brier = mean
                best_k = k
                best_shrink = shrink
                marker = " <<<" if mean < old_mean else ""
            else:
                marker = ""

            # Only print interesting results
            if abs(delta) < 0.005 or mean < old_mean:
                print(f"{k:6.1f} {shrink:8.2f} {mean:8.4f} {delta:+10.4f} {is_better}{marker}")

    print("-" * 44)
    print(f"\nBEST: K={best_k:.1f}, Shrinkage={best_shrink:.2f}, Brier={best_brier:.4f}")
    delta = best_brier - old_mean
    print(f"  vs old model: {delta:+.4f} ({'IMPROVED' if delta < 0 else 'WORSE'})")

    # Also test without clutch and without L10 to isolate their contribution
    print("\n--- Ablation study (at best params) ---")
    for label, use_l10, use_clutch in [
        ("Full (season+L10+clutch)", True, True),
        ("Season + L10 only", True, False),
        ("Season only", False, False),
    ]:
        total = 0.0
        for g in nba_games:
            h_s = season[g["home_name"]]
            a_s = season[g["away_name"]]
            h_l = l10.get(g["home_name"]) if use_l10 else None
            a_l = l10.get(g["away_name"]) if use_l10 else None
            h_c = clutch.get(g["home_name"]) if use_clutch else None
            a_c = clutch.get(g["away_name"]) if use_clutch else None

            prob = net_rtg_probability(
                h_s["net_rtg"], a_s["net_rtg"],
                h_l["net_rtg"] if h_l else None,
                a_l["net_rtg"] if a_l else None,
                h_c["net_rtg"] if h_c else None,
                a_c["net_rtg"] if a_c else None,
                k=best_k, shrinkage=best_shrink,
            )
            total += brier(prob, g["actual"])
        mean = total / n
        print(f"  {label:35s}: Brier={mean:.4f} (Δ {mean - old_mean:+.4f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=str(PROJECT_DIR / "backtest_results" / "nba_nhl_backtest_20260323_015239.csv"))
    args = parser.parse_args()
    run(args.csv)
