import csv

from scripts.tennis_backtest_runner import run_backtest
from src.services.tennis_prediction import build_tennis_prediction_record, predict_tennis


def _match() -> dict:
    return {
        "id": "atp-1",
        "tour": "ATP",
        "player_a_id": "sr:competitor:1",
        "player_b_id": "sr:competitor:2",
        "player_a": "Player A",
        "player_b": "Player B",
        "tournament": "Miami Open",
        "round_name": "Quarterfinal",
        "surface": "hard",
        "best_of": 3,
        "scheduled_time": "2026-03-29T18:00:00Z",
        "indoor": False,
    }


def _match_data() -> dict:
    return {
        "player_a": {
            "rating": 1820,
            "surface_rating": 1840,
            "recent_form": 0.72,
            "hold_pct": 0.84,
            "break_pct": 0.25,
            "injury_risk": 0.03,
        },
        "player_b": {
            "rating": 1710,
            "surface_rating": 1690,
            "recent_form": 0.55,
            "hold_pct": 0.78,
            "break_pct": 0.19,
            "injury_risk": 0.10,
        },
        "context": {
            "player_a_rest_days": 2,
            "player_b_rest_days": 1,
            "player_a_last_match_minutes": 90,
            "player_b_last_match_minutes": 150,
            "player_a_travel_zones": 0,
            "player_b_travel_zones": 1,
        },
        "h2h": {
            "player_a_win_pct": 0.67,
            "sample": 3,
        },
        "market": {
            "player_a_price": 0.58,
            "player_b_price": 0.46,
        },
    }


def test_build_tennis_prediction_record_contains_tennis_metadata():
    artifacts = predict_tennis(_match(), _match_data())
    record = build_tennis_prediction_record(None, artifacts)

    assert record.sport == "TENNIS"
    assert record.model_family == "tennis_rule_model"
    assert record.snapshot_metadata["tour"] == "ATP"
    assert record.feature_snapshot["tour"] == "ATP"


def test_tennis_backtest_runner_processes_csv(tmp_path):
    csv_path = tmp_path / "tennis_backtest.csv"
    fieldnames = [
        "match_id",
        "tour",
        "player_a",
        "player_b",
        "tournament",
        "round_name",
        "surface",
        "best_of",
        "scheduled_time",
        "indoor",
        "player_a_rating",
        "player_b_rating",
        "player_a_surface_rating",
        "player_b_surface_rating",
        "player_a_recent_form",
        "player_b_recent_form",
        "player_a_hold_pct",
        "player_b_hold_pct",
        "player_a_break_pct",
        "player_b_break_pct",
        "player_a_injury_risk",
        "player_b_injury_risk",
        "player_a_rest_days",
        "player_b_rest_days",
        "player_a_last_match_minutes",
        "player_b_last_match_minutes",
        "player_a_travel_zones",
        "player_b_travel_zones",
        "h2h_player_a_win_pct",
        "h2h_sample",
        "market_player_a_price",
        "market_player_b_price",
        "player_a_won",
    ]
    rows = [
        {
            "match_id": "m1",
            "tour": "ATP",
            "player_a": "Player A",
            "player_b": "Player B",
            "tournament": "Miami Open",
            "round_name": "Quarterfinal",
            "surface": "hard",
            "best_of": 3,
            "scheduled_time": "2026-03-29T18:00:00Z",
            "indoor": False,
            "player_a_rating": 1820,
            "player_b_rating": 1710,
            "player_a_surface_rating": 1840,
            "player_b_surface_rating": 1690,
            "player_a_recent_form": 0.72,
            "player_b_recent_form": 0.55,
            "player_a_hold_pct": 0.84,
            "player_b_hold_pct": 0.78,
            "player_a_break_pct": 0.25,
            "player_b_break_pct": 0.19,
            "player_a_injury_risk": 0.03,
            "player_b_injury_risk": 0.10,
            "player_a_rest_days": 2,
            "player_b_rest_days": 1,
            "player_a_last_match_minutes": 90,
            "player_b_last_match_minutes": 150,
            "player_a_travel_zones": 0,
            "player_b_travel_zones": 1,
            "h2h_player_a_win_pct": 0.67,
            "h2h_sample": 3,
            "market_player_a_price": 0.58,
            "market_player_b_price": 0.46,
            "player_a_won": True,
        },
        {
            "match_id": "m2",
            "tour": "WTA",
            "player_a": "Player C",
            "player_b": "Player D",
            "tournament": "Madrid Open",
            "round_name": "Semifinal",
            "surface": "clay",
            "best_of": 3,
            "scheduled_time": "2026-03-30T18:00:00Z",
            "indoor": False,
            "player_a_rating": 1750,
            "player_b_rating": 1780,
            "player_a_surface_rating": 1800,
            "player_b_surface_rating": 1760,
            "player_a_recent_form": 0.68,
            "player_b_recent_form": 0.63,
            "player_a_hold_pct": 0.73,
            "player_b_hold_pct": 0.75,
            "player_a_break_pct": 0.28,
            "player_b_break_pct": 0.24,
            "player_a_injury_risk": 0.04,
            "player_b_injury_risk": 0.02,
            "player_a_rest_days": 2,
            "player_b_rest_days": 2,
            "player_a_last_match_minutes": 105,
            "player_b_last_match_minutes": 118,
            "player_a_travel_zones": 0,
            "player_b_travel_zones": 0,
            "h2h_player_a_win_pct": 0.5,
            "h2h_sample": 2,
            "market_player_a_price": 0.49,
            "market_player_b_price": 0.53,
            "player_a_won": False,
        },
    ]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    result = run_backtest(csv_path)

    assert result["matches"] == 2
    assert result["brier"] >= 0
    assert result["ending_bankroll"] > 0
