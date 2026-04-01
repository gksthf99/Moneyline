from __future__ import annotations

from src.domain.prediction_record import GradingRecord, PredictionRecord
from src.features.game_features import FeatureSnapshot
from src.features.model_features import (
    build_nba_model_features,
    build_nhl_model_features,
)


def build_ensemble_training_row(
    prediction: PredictionRecord,
    grade: GradingRecord,
) -> dict:
    feature = prediction.feature_snapshot or {}
    home_net = (feature.get("home_net_rtg") or {}).get("season", {}) or {}
    away_net = (feature.get("away_net_rtg") or {}).get("season", {}) or {}

    return {
        "sport": prediction.sport,
        "model_variant": prediction.model_variant,
        "base_probability": prediction.base_probability,
        "final_probability": prediction.final_probability,
        "net_rating_spread": round(home_net.get("net_rtg", 0.0) - away_net.get("net_rtg", 0.0), 3),
        "player_impact_adj": round(feature.get("player_impact_adj", prediction.information_edge), 4),
        "rest_differential": round(feature.get("rest_differential", 0.0), 3),
        "home_court": 1.0,
        "pace_differential": round(home_net.get("pace", 100.0) - away_net.get("pace", 100.0), 3),
        "market_price": feature.get("market_home_price"),
        "goalie_adjustment": round(feature.get("goalie_adjustment", 0.0), 4),
        "clv_residual": feature.get("clv_residual"),
        "home_won": grade.home_won,
    }


def build_replacement_model_training_row(
    prediction: PredictionRecord,
    grade: GradingRecord,
) -> dict:
    feature_payload = dict(prediction.feature_snapshot or {})
    feature_payload.setdefault("sport", prediction.sport)
    feature_payload.setdefault("home_team", prediction.home_team)
    feature_payload.setdefault("away_team", prediction.away_team)
    feature = FeatureSnapshot(**feature_payload)
    sport = prediction.sport.upper()

    if sport == "NBA":
        model_features = build_nba_model_features(
            feature,
            base_probability=prediction.base_probability,
            final_probability=prediction.final_probability,
        )
    elif sport == "NHL":
        model_features = build_nhl_model_features(
            feature,
            base_probability=prediction.base_probability,
            final_probability=prediction.final_probability,
        )
    else:
        raise ValueError(f"Unsupported sport for replacement model row: {prediction.sport}")

    row = model_features.to_training_row()
    row.update(
        {
            "model_variant": prediction.model_variant,
            "recommendation": prediction.recommendation,
            "effective_edge": prediction.effective_edge,
            "home_won": grade.home_won,
        }
    )
    return row
