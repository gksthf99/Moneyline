from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.tennis_api import build_match_data, get_schedule
from src.repositories.snapshot_repository import SnapshotRepository
from src.services.tennis_prediction import persist_tennis_prediction, predict_tennis


def build_research_rows(
    match_date: date | None = None,
    *,
    tour: str | None = None,
    limit: int | None = None,
    persist: bool = False,
    repository: SnapshotRepository | None = None,
) -> list[dict]:
    rows: list[dict] = []
    matches = get_schedule(match_date, tour=tour)
    if limit is not None:
        matches = matches[:limit]

    for match in matches:
        match_data = build_match_data(match)
        artifacts = predict_tennis(match, match_data)
        record = persist_tennis_prediction(artifacts, repository=repository) if persist else None
        rows.append(
            {
                "match_id": artifacts.snapshot.match_id,
                "tour": artifacts.snapshot.tour,
                "tournament": artifacts.snapshot.tournament,
                "round_name": artifacts.snapshot.round_name,
                "scheduled_time": artifacts.snapshot.scheduled_time,
                "player_a": artifacts.snapshot.player_a,
                "player_b": artifacts.snapshot.player_b,
                "final_probability_player_a": round(artifacts.decomposition.final_probability, 4),
                "recommendation": artifacts.recommendation,
                "summary": artifacts.summary,
                "prediction_record_created": record is not None,
            }
        )
    return rows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch and score a tennis slate.")
    parser.add_argument("--date", type=str, default=None, help="Slate date in YYYY-MM-DD format")
    parser.add_argument("--tour", type=str, default=None, help="ATP or WTA")
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of matches")
    parser.add_argument("--persist", action="store_true", help="Persist snapshots and prediction records")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of plain summaries")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    match_date = date.fromisoformat(args.date) if args.date else None
    rows = build_research_rows(
        match_date,
        tour=args.tour,
        limit=args.limit,
        persist=args.persist,
    )
    if args.json:
        print(json.dumps(rows, indent=2))
        return 0
    for row in rows:
        print(row["summary"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
