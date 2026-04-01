from src.domain.prediction_record import GradingRecord, PredictionRecord
from src.services.training_dataset import (
    build_ensemble_training_row,
    build_replacement_model_training_row,
)


def test_build_ensemble_training_row_from_prediction_and_grade():
    prediction = PredictionRecord(
        prediction_id=1,
        game_id=2,
        sport="NBA",
        home_team="Bulls",
        away_team="Knicks",
        game_time="2026-03-29T23:00:00Z",
        triage_level="standard",
        base_probability=0.57,
        situational_adjustment=0.01,
        information_edge=-0.02,
        final_probability=0.56,
        edge_type="B",
        recommendation="PASS",
        effective_edge=0.01,
        bet_side=None,
        feature_snapshot={
            "home_net_rtg": {"season": {"net_rtg": 4.2, "pace": 99.5}},
            "away_net_rtg": {"season": {"net_rtg": 1.2, "pace": 97.0}},
            "market_home_price": 0.58,
            "rest_differential": 1.0,
            "goalie_adjustment": 0.0,
            "player_impact_adj": -0.02,
            "clv_residual": 0.01,
        },
    )
    grade = GradingRecord(
        prediction_id=1,
        game_id=2,
        home_team="Bulls",
        away_team="Knicks",
        predicted_home_prob=0.56,
        home_won=True,
        home_score=110,
        away_score=101,
        brier_score=0.1936,
        edge_type="B",
        effective_edge=0.01,
        recommendation="PASS",
        base_correct=True,
        sit_helped=True,
        info_helped=False,
    )

    row = build_ensemble_training_row(prediction, grade)

    assert row["net_rating_spread"] == 3.0
    assert row["pace_differential"] == 2.5
    assert row["player_impact_adj"] == -0.02
    assert row["rest_differential"] == 1.0
    assert row["clv_residual"] == 0.01
    assert row["market_price"] == 0.58
    assert row["home_won"] is True


def test_build_replacement_model_training_row_from_prediction_and_grade():
    prediction = PredictionRecord(
        prediction_id=1,
        game_id=2,
        sport="NHL",
        home_team="Stars",
        away_team="Jets",
        game_time="2026-03-29T23:00:00Z",
        triage_level="standard",
        base_probability=0.57,
        situational_adjustment=0.01,
        information_edge=0.02,
        final_probability=0.60,
        edge_type="B",
        recommendation="BET",
        effective_edge=0.03,
        bet_side="home",
        feature_snapshot={
            "home_win_pct": 0.61,
            "away_win_pct": 0.57,
            "home_home_pct": 0.65,
            "away_away_pct": 0.5,
            "home_l10_pct": 0.6,
            "away_l10_pct": 0.5,
            "home_xgf_pct": 0.54,
            "away_xgf_pct": 0.5,
            "home_gf_ga_ratio": 0.56,
            "away_gf_ga_ratio": 0.51,
            "rest_differential": 1.0,
            "home_is_b2b": False,
            "away_is_b2b": True,
            "away_travel_zones": 1,
            "market_home_price": 0.55,
            "goalie_adjustment": 0.03,
            "player_impact_adj": 0.0,
            "injury_count": 1,
            "clv_residual": -0.01,
        },
    )
    grade = GradingRecord(
        prediction_id=1,
        game_id=2,
        home_team="Stars",
        away_team="Jets",
        predicted_home_prob=0.60,
        home_won=True,
        home_score=4,
        away_score=2,
        brier_score=0.16,
        edge_type="B",
        effective_edge=0.03,
        recommendation="BET",
        base_correct=True,
        sit_helped=True,
        info_helped=True,
    )

    row = build_replacement_model_training_row(prediction, grade)

    assert row["sport"] == "NHL"
    assert row["xgf_pct_spread"] == 0.04
    assert row["gf_ga_ratio_spread"] == 0.05
    assert row["goalie_adjustment"] == 0.03
    assert row["market_home_probability"] == 0.55
    assert row["home_won"] is True
