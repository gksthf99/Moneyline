#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from src.model.edge import calculate_position_size
from src.services.tennis_prediction import predict_tennis


def _bool(value: str | None) -> bool:
    return str(value).strip().lower() in {"1", "true", "t", "yes"}


def _float(value: str | None, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    return float(value)


def _int(value: str | None, default: int = 0) -> int:
    if value in (None, ""):
        return default
    return int(float(value))


def _build_match_and_data(row: dict) -> tuple[dict, dict, bool]:
    match = {
        "id": row.get("match_id") or row.get("id"),
        "tour": row.get("tour", "ATP"),
        "player_a_id": row.get("player_a_id"),
        "player_b_id": row.get("player_b_id"),
        "player_a": row.get("player_a", ""),
        "player_b": row.get("player_b", ""),
        "tournament": row.get("tournament", ""),
        "round_name": row.get("round_name", ""),
        "surface": row.get("surface", "hard"),
        "best_of": _int(row.get("best_of"), 3),
        "scheduled_time": row.get("scheduled_time", row.get("match_date", "")),
        "indoor": _bool(row.get("indoor")),
        "player_a_rest_days": _int(row.get("player_a_rest_days"), 1),
        "player_b_rest_days": _int(row.get("player_b_rest_days"), 1),
        "player_a_last_match_minutes": _int(row.get("player_a_last_match_minutes"), 0),
        "player_b_last_match_minutes": _int(row.get("player_b_last_match_minutes"), 0),
        "player_a_travel_zones": _int(row.get("player_a_travel_zones"), 0),
        "player_b_travel_zones": _int(row.get("player_b_travel_zones"), 0),
        "h2h_player_a_win_pct": _float(row.get("h2h_player_a_win_pct"), 0.0) if row.get("h2h_player_a_win_pct") not in (None, "") else None,
        "h2h_sample": _int(row.get("h2h_sample"), 0),
    }
    match_data = {
        "player_a": {
            "rating": _float(row.get("player_a_rating"), 1500.0),
            "surface_rating": _float(row.get("player_a_surface_rating"), _float(row.get("player_a_rating"), 1500.0)),
            "recent_form": _float(row.get("player_a_recent_form"), 0.5),
            "hold_pct": _float(row.get("player_a_hold_pct"), 0.75),
            "break_pct": _float(row.get("player_a_break_pct"), 0.22),
            "injury_risk": _float(row.get("player_a_injury_risk"), 0.0),
        },
        "player_b": {
            "rating": _float(row.get("player_b_rating"), 1500.0),
            "surface_rating": _float(row.get("player_b_surface_rating"), _float(row.get("player_b_rating"), 1500.0)),
            "recent_form": _float(row.get("player_b_recent_form"), 0.5),
            "hold_pct": _float(row.get("player_b_hold_pct"), 0.75),
            "break_pct": _float(row.get("player_b_break_pct"), 0.22),
            "injury_risk": _float(row.get("player_b_injury_risk"), 0.0),
        },
        "context": {
            "player_a_rest_days": match["player_a_rest_days"],
            "player_b_rest_days": match["player_b_rest_days"],
            "player_a_last_match_minutes": match["player_a_last_match_minutes"],
            "player_b_last_match_minutes": match["player_b_last_match_minutes"],
            "player_a_travel_zones": match["player_a_travel_zones"],
            "player_b_travel_zones": match["player_b_travel_zones"],
        },
        "h2h": {
            "player_a_win_pct": match["h2h_player_a_win_pct"],
            "sample": match["h2h_sample"],
        },
        "market": {
            "player_a_price": _float(row.get("market_player_a_price"), None) if row.get("market_player_a_price") not in (None, "") else None,
            "player_b_price": _float(row.get("market_player_b_price"), None) if row.get("market_player_b_price") not in (None, "") else None,
        },
    }
    player_a_won = _bool(row.get("player_a_won"))
    return match, match_data, player_a_won


def run_backtest(csv_path: Path, starting_bankroll: float = 1000.0) -> dict:
    rows = []
    with csv_path.open() as handle:
        for row in csv.DictReader(handle):
            match, match_data, player_a_won = _build_match_and_data(row)
            artifacts = predict_tennis(match, match_data)
            rows.append((artifacts, player_a_won))

    bankroll = starting_bankroll
    peak = bankroll
    max_drawdown = 0.0
    total_brier = 0.0
    wins = losses = bets = 0

    for artifacts, player_a_won in rows:
        prob = artifacts.decomposition.final_probability
        actual = 1.0 if player_a_won else 0.0
        total_brier += (prob - actual) ** 2

        recommendation = artifacts.recommendation
        favorite_is_a = prob >= 0.5
        market_price = (
            artifacts.feature_snapshot.market_player_a_price
            if favorite_is_a
            else artifacts.feature_snapshot.market_player_b_price
        )
        if recommendation == "BET" and market_price:
            model_side_prob = max(prob, 1.0 - prob)
            stake = calculate_position_size(
                model_prob=model_side_prob,
                ask_price=float(market_price),
                bankroll=bankroll,
                edge_type="B",
                sport="TENNIS",
            )
            if stake > 0:
                won = player_a_won if favorite_is_a else (not player_a_won)
                shares = stake / float(market_price)
                pnl = (shares - stake) if won else -stake
                bankroll += pnl
                peak = max(peak, bankroll)
                max_drawdown = max(max_drawdown, (peak - bankroll) / peak if peak else 0.0)
                bets += 1
                wins += int(won)
                losses += int(not won)

    return {
        "matches": len(rows),
        "bets": bets,
        "wins": wins,
        "losses": losses,
        "win_rate": (wins / bets) if bets else 0.0,
        "brier": (total_brier / len(rows)) if rows else 0.0,
        "ending_bankroll": bankroll,
        "profit": bankroll - starting_bankroll,
        "max_drawdown": max_drawdown,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a tennis backtest from a historical CSV.")
    parser.add_argument("--csv", required=True, help="Historical tennis match CSV")
    parser.add_argument("--bankroll", type=float, default=1000.0, help="Starting bankroll")
    args = parser.parse_args()

    result = run_backtest(Path(args.csv), starting_bankroll=args.bankroll)
    print(f"Matches: {result['matches']}")
    print(f"Bets: {result['bets']}")
    print(f"Record: {result['wins']}-{result['losses']} ({result['win_rate']:.1%})")
    print(f"Brier: {result['brier']:.4f}")
    print(f"Ending bankroll: ${result['ending_bankroll']:.2f}")
    print(f"Profit: ${result['profit']:.2f}")
    print(f"Max drawdown: {result['max_drawdown']:.1%}")


if __name__ == "__main__":
    main()
