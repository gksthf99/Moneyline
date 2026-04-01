#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import date, timedelta
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from src.repositories.prediction_record_repository import PredictionRecordRepository
from src.repositories.snapshot_repository import RECORD_LOG


def _read_local_prediction_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows
def export_from_local_lineage(path: Path) -> list[dict]:
    return _read_local_prediction_records(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export replacement-model training features from persisted prediction artifacts.")
    parser.add_argument("--start", help="Start date YYYY-MM-DD for Supabase prediction records")
    parser.add_argument("--end", help="End date YYYY-MM-DD")
    parser.add_argument("--source", choices=("supabase", "local"), default="local")
    parser.add_argument("--input", help="Local JSONL lineage file", default=str(RECORD_LOG))
    parser.add_argument("--output", required=True, help="Output CSV path")
    args = parser.parse_args()

    rows: list[dict]
    if args.source == "supabase":
        if not args.start:
            raise SystemExit("--start is required when --source supabase")
        start = date.fromisoformat(args.start)
        end = date.fromisoformat(args.end) if args.end else (start + timedelta(days=1))
        records = PredictionRecordRepository().list_records(start, end)
        rows = []
        for record in records:
            payload = dict(record.feature_snapshot or {})
            payload.update(
                {
                    "sport": record.sport,
                    "home_team": record.home_team,
                    "away_team": record.away_team,
                    "base_probability": record.base_probability,
                    "final_probability": record.final_probability,
                    "model_variant": record.model_variant,
                }
            )
            rows.append(payload)
    else:
        rows = export_from_local_lineage(Path(args.input))

    if not rows:
        raise SystemExit("No rows found to export")

    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(args.output, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
