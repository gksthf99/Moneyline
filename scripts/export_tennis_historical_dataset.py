#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import sys
from datetime import date, timedelta
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from src.data.tennis_api import build_match_data, get_results, get_schedule


def _daterange(start: date, end: date):
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def _row_from_match(match: dict, match_data: dict, player_a_won: bool | None) -> dict:
    player_a = match_data.get("player_a") or {}
    player_b = match_data.get("player_b") or {}
    context = match_data.get("context") or {}
    market = match_data.get("market") or {}
    h2h = match_data.get("h2h") or {}
    return {
        "match_id": match.get("id"),
        "tour": match.get("tour"),
        "player_a_id": match.get("player_a_id"),
        "player_b_id": match.get("player_b_id"),
        "player_a": match.get("player_a"),
        "player_b": match.get("player_b"),
        "tournament": match.get("tournament"),
        "round_name": match.get("round_name"),
        "surface": match.get("surface"),
        "best_of": match.get("best_of"),
        "scheduled_time": match.get("scheduled_time"),
        "indoor": match.get("indoor"),
        "status": match.get("status"),
        "player_a_rating": player_a.get("rating"),
        "player_b_rating": player_b.get("rating"),
        "player_a_surface_rating": player_a.get("surface_rating"),
        "player_b_surface_rating": player_b.get("surface_rating"),
        "player_a_recent_form": player_a.get("recent_form"),
        "player_b_recent_form": player_b.get("recent_form"),
        "player_a_hold_pct": player_a.get("hold_pct"),
        "player_b_hold_pct": player_b.get("hold_pct"),
        "player_a_break_pct": player_a.get("break_pct"),
        "player_b_break_pct": player_b.get("break_pct"),
        "player_a_injury_risk": player_a.get("injury_risk"),
        "player_b_injury_risk": player_b.get("injury_risk"),
        "player_a_rest_days": context.get("player_a_rest_days"),
        "player_b_rest_days": context.get("player_b_rest_days"),
        "player_a_last_match_minutes": context.get("player_a_last_match_minutes"),
        "player_b_last_match_minutes": context.get("player_b_last_match_minutes"),
        "player_a_travel_zones": context.get("player_a_travel_zones"),
        "player_b_travel_zones": context.get("player_b_travel_zones"),
        "h2h_player_a_win_pct": h2h.get("player_a_win_pct"),
        "h2h_sample": h2h.get("sample"),
        "market_player_a_price": market.get("player_a_price"),
        "market_player_b_price": market.get("player_b_price"),
        "player_a_won": player_a_won,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export historical tennis matches into the backtest CSV schema.")
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--tour", choices=["ATP", "WTA"], help="Optional tour filter")
    parser.add_argument("--completed-only", action="store_true", help="Export completed matches only")
    parser.add_argument("--output", required=True, help="Output CSV path")
    args = parser.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)

    rows: list[dict] = []
    for current in _daterange(start, end):
        matches = get_results(current, tour=args.tour) if args.completed_only else get_schedule(current, tour=args.tour)
        for match in matches:
            if args.completed_only and (match.get("status") or "").lower() not in {"closed", "ended", "complete", "finished"}:
                continue
            match_data = build_match_data(match)
            player_a_won = None
            rows.append(_row_from_match(match, match_data, player_a_won))

    if not rows:
        raise SystemExit("No tennis matches exported. Check provider config or date range.")

    fieldnames = list(rows[0].keys())
    with open(args.output, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
