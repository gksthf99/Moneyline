#!/usr/bin/env python3

import argparse
import csv
from datetime import date, timedelta

from src.repositories.prediction_record_repository import PredictionRecordRepository


def main():
    parser = argparse.ArgumentParser(description="Export persisted prediction records as a training dataset.")
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", help="End date YYYY-MM-DD")
    parser.add_argument("--output", required=True, help="Output CSV path")
    args = parser.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end) if args.end else (start + timedelta(days=1))

    repo = PredictionRecordRepository()
    rows = repo.list_records(start, end)

    with open(args.output, "w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "game_id",
                "sport",
                "home_team",
                "away_team",
                "game_time",
                "triage_level",
                "base_probability",
                "situational_adjustment",
                "information_edge",
                "final_probability",
                "rest_differential",
                "goalie_adjustment",
                "player_impact_adj",
                "clv_residual",
                "edge_type",
                "recommendation",
                "effective_edge",
                "bet_side",
                "model_family",
                "model_variant",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "game_id": row.game_id,
                "sport": row.sport,
                "home_team": row.home_team,
                "away_team": row.away_team,
                "game_time": row.game_time,
                "triage_level": row.triage_level,
                "base_probability": row.base_probability,
                "situational_adjustment": row.situational_adjustment,
                "information_edge": row.information_edge,
                "final_probability": row.final_probability,
                "rest_differential": row.feature_snapshot.get("rest_differential"),
                "goalie_adjustment": row.feature_snapshot.get("goalie_adjustment"),
                "player_impact_adj": row.feature_snapshot.get("player_impact_adj"),
                "clv_residual": row.feature_snapshot.get("clv_residual"),
                "edge_type": row.edge_type,
                "recommendation": row.recommendation,
                "effective_edge": row.effective_edge,
                "bet_side": row.bet_side,
                "model_family": row.model_family,
                "model_variant": row.model_variant,
            })


if __name__ == "__main__":
    main()
