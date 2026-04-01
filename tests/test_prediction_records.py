import json

from src.agents.performance_agent import grade_game
from src.domain.prediction_record import PredictionRecord


def test_prediction_record_from_supabase_row():
    row = {
        "id": 10,
        "game_id": 22,
        "prob_decomposition": json.dumps({
            "base": 0.57,
            "situational": 0.01,
            "information": -0.02,
            "final": 0.56,
            "bet_side": "home",
        }),
        "edge_type": "C",
        "effective_edge": 0.034,
        "recommendation": "BET",
        "created_at": "2026-03-29T12:00:00Z",
        "data_freshness_status": "model_variant=experimental_ensemble",
        "feature_snapshot": json.dumps({"market_home_price": 0.58}),
        "snapshot_metadata": json.dumps({"collected_at": "2026-03-29T11:30:00Z"}),
    }
    game = {
        "sport": "NBA",
        "home_team": "Bulls",
        "away_team": "Knicks",
        "game_time": "2026-03-29T23:00:00Z",
        "triage_level": "standard",
    }

    record = PredictionRecord.from_supabase_row(row, game)

    assert record is not None
    assert record.prediction_id == 10
    assert record.final_probability == 0.56
    assert record.model_variant == "experimental_ensemble"
    assert record.home_team == "Bulls"
    assert record.feature_snapshot["market_home_price"] == 0.58


def test_grade_game_uses_typed_prediction_record():
    prediction = PredictionRecord(
        prediction_id=10,
        game_id=22,
        sport="NBA",
        home_team="Bulls",
        away_team="Knicks",
        game_time="2026-03-29T23:00:00Z",
        triage_level="standard",
        base_probability=0.57,
        situational_adjustment=0.01,
        information_edge=-0.02,
        final_probability=0.56,
        edge_type="C",
        recommendation="BET",
        effective_edge=0.034,
        bet_side="home",
    )
    outcome = {
        "home_team": "Bulls",
        "away_team": "Knicks",
        "home_score": 110,
        "away_score": 101,
        "home_won": True,
    }

    grade = grade_game(prediction, outcome)

    assert grade is not None
    assert grade.brier_score == round((0.56 - 1) ** 2, 6)
    assert grade.base_correct is True
    assert grade.info_helped is False
