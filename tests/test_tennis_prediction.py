from src.domain.tennis_match import TennisMatchSnapshot
from src.features.tennis_features import build_tennis_feature_snapshot
from src.model.rebuild_spec import get_replacement_model_spec
from src.model.tennis_model import predict_tennis_match
from src.services.tennis_prediction import classify_tennis_recommendation, predict_tennis


def _sample_match() -> dict:
    return {
        "id": "atp-miami-sf-a-b",
        "tour": "ATP",
        "player_a_id": "sr:competitor:1",
        "player_b_id": "sr:competitor:2",
        "player_a": "Player A",
        "player_b": "Player B",
        "tournament": "Miami Open",
        "round_name": "Semifinal",
        "surface": "hard",
        "best_of": 3,
        "scheduled_time": "2026-03-29T18:00:00Z",
        "indoor": False,
    }


def _sample_match_data() -> dict:
    return {
        "player_a": {
            "rating": 1850,
            "surface_rating": 1880,
            "recent_form": 0.78,
            "hold_pct": 0.84,
            "break_pct": 0.24,
            "injury_risk": 0.05,
        },
        "player_b": {
            "rating": 1765,
            "surface_rating": 1730,
            "recent_form": 0.58,
            "hold_pct": 0.79,
            "break_pct": 0.19,
            "injury_risk": 0.12,
        },
        "context": {
            "player_a_rest_days": 2,
            "player_b_rest_days": 1,
            "player_a_last_match_minutes": 95,
            "player_b_last_match_minutes": 168,
            "player_a_travel_zones": 0,
            "player_b_travel_zones": 2,
        },
        "h2h": {
            "player_a_win_pct": 0.67,
            "sample": 3,
        },
        "market": {
            "player_a_price": 0.60,
            "player_b_price": 0.42,
        },
    }


def test_build_tennis_feature_snapshot_extracts_match_state():
    snapshot = TennisMatchSnapshot.from_match_and_data(_sample_match(), _sample_match_data())
    feature = build_tennis_feature_snapshot(snapshot, _sample_match_data())

    assert feature.player_a_rating == 1850
    assert feature.player_b_surface_rating == 1730
    assert feature.player_b_last_match_minutes == 168
    assert feature.market_player_a_price == 0.60
    assert snapshot.player_a_id == "sr:competitor:1"


def test_predict_tennis_match_favors_stronger_player():
    snapshot = TennisMatchSnapshot.from_match_and_data(_sample_match(), _sample_match_data())
    feature = build_tennis_feature_snapshot(snapshot, _sample_match_data())
    decomposition = predict_tennis_match(feature)

    assert decomposition.base_probability > 0.5
    assert decomposition.final_probability > decomposition.base_probability
    assert decomposition.final_probability > 0.60


def test_classify_tennis_recommendation_uses_market_gap():
    snapshot = TennisMatchSnapshot.from_match_and_data(_sample_match(), _sample_match_data())
    feature = build_tennis_feature_snapshot(snapshot, _sample_match_data())
    decomposition = predict_tennis_match(feature)

    assert classify_tennis_recommendation(decomposition, feature) == "BET"


def test_predict_tennis_builds_summary_and_variant():
    artifacts = predict_tennis(_sample_match(), _sample_match_data())

    assert artifacts.model_variant == "tennis_match_v1"
    assert "ATP" in artifacts.summary
    assert "Miami Open" in artifacts.summary
    assert artifacts.recommendation in {"BET", "PASS", "MONITOR"}


def test_tennis_replacement_spec_is_registered():
    spec = get_replacement_model_spec("TENNIS")

    assert spec.model_name == "tennis_point_in_time_logistic_v1"
    assert "player_a_surface_rating" in spec.feature_names
