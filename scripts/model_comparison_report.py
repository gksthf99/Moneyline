#!/usr/bin/env python3

from collections import defaultdict
from datetime import date
import argparse

from src.agents.performance_agent import get_resolved_nba, get_resolved_nhl
from src.repositories.prediction_record_repository import PredictionRecordRepository


def _fuzzy_match(name_a: str, name_b: str) -> bool:
    if name_a == name_b:
        return True
    if not name_a or not name_b:
        return False
    return name_a.split()[-1].lower() == name_b.split()[-1].lower()


def main():
    parser = argparse.ArgumentParser(description="Compare model variants using persisted prediction records.")
    parser.add_argument("--date", required=True, help="Sports date YYYY-MM-DD")
    args = parser.parse_args()
    target = date.fromisoformat(args.date)

    repo = PredictionRecordRepository()
    records = repo.list_records(target)
    outcomes = get_resolved_nba(target) + get_resolved_nhl(target)

    stats = defaultdict(lambda: {"n": 0, "brier": 0.0, "correct": 0})
    for record in records:
        for outcome in outcomes:
            if _fuzzy_match(record.home_team, outcome["home_team"]) and _fuzzy_match(record.away_team, outcome["away_team"]):
                actual = 1.0 if outcome["home_won"] else 0.0
                brier = (record.final_probability - actual) ** 2
                stats[record.model_variant]["n"] += 1
                stats[record.model_variant]["brier"] += brier
                if (record.final_probability >= 0.5) == outcome["home_won"]:
                    stats[record.model_variant]["correct"] += 1
                break

    print(f"Model comparison for {target.isoformat()}")
    print(f"{'Variant':24s} {'N':>4s} {'Brier':>8s} {'Accuracy':>9s}")
    for variant, s in sorted(stats.items()):
        n = s["n"]
        if n == 0:
            continue
        print(f"{variant:24s} {n:4d} {s['brier']/n:8.4f} {s['correct']/n:9.1%}")


if __name__ == "__main__":
    main()
