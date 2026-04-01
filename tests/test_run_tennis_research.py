from __future__ import annotations

from datetime import date

from scripts.run_tennis_research import build_research_rows


def _sample_match() -> dict:
    return {
        "id": "match-1",
        "tour": "ATP",
        "player_a": "Carlos Alcaraz",
        "player_b": "Jannik Sinner",
        "tournament": "Miami Open",
        "round_name": "Semifinal",
        "surface": "hard",
        "best_of": 3,
        "scheduled_time": "2026-03-29T18:00:00Z",
        "player_a_id": "a1",
        "player_b_id": "b1",
    }


def _sample_match_data() -> dict:
    return {
        "player_a": {
            "rating": 1760,
            "surface_rating": 1780,
            "recent_form": 0.72,
            "hold_pct": 0.86,
            "break_pct": 0.28,
            "injury_risk": 0.02,
        },
        "player_b": {
            "rating": 1730,
            "surface_rating": 1750,
            "recent_form": 0.69,
            "hold_pct": 0.84,
            "break_pct": 0.25,
            "injury_risk": 0.04,
        },
        "context": {
            "player_a_rest_days": 2,
            "player_b_rest_days": 1,
            "player_a_last_match_minutes": 95,
            "player_b_last_match_minutes": 145,
            "player_a_travel_zones": 0,
            "player_b_travel_zones": 1,
        },
        "h2h": {"player_a_win_pct": 0.6, "sample": 5},
        "market": {"player_a_price": 0.58, "player_b_price": 0.42},
    }


def test_build_research_rows(monkeypatch) -> None:
    monkeypatch.setattr(
        "scripts.run_tennis_research.get_schedule",
        lambda match_date, tour=None: [_sample_match()],
    )
    monkeypatch.setattr(
        "scripts.run_tennis_research.build_match_data",
        lambda match: _sample_match_data(),
    )

    rows = build_research_rows(date(2026, 3, 29), tour="ATP")

    assert len(rows) == 1
    row = rows[0]
    assert row["match_id"] == "match-1"
    assert row["tour"] == "ATP"
    assert row["player_a"] == "Carlos Alcaraz"
    assert row["player_b"] == "Jannik Sinner"
    assert row["recommendation"] in {"BET", "PASS", "MONITOR"}
    assert row["prediction_record_created"] is False
    assert isinstance(row["summary"], str)


def test_build_research_rows_with_persistence(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        "scripts.run_tennis_research.get_schedule",
        lambda match_date, tour=None: [_sample_match()],
    )
    monkeypatch.setattr(
        "scripts.run_tennis_research.build_match_data",
        lambda match: _sample_match_data(),
    )

    from src.repositories.snapshot_repository import SnapshotRepository

    repository = SnapshotRepository(
        snapshot_path=tmp_path / "prediction_snapshots.jsonl",
        record_path=tmp_path / "prediction_records.jsonl",
    )

    rows = build_research_rows(date(2026, 3, 29), persist=True, repository=repository)

    assert rows[0]["prediction_record_created"] is True
    assert (tmp_path / "prediction_snapshots.jsonl").exists()
    assert (tmp_path / "prediction_records.jsonl").exists()
