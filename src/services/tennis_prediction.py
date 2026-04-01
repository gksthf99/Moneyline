from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from src.domain.prediction_record import PredictionRecord
from src.domain.tennis_match import TennisMatchSnapshot
from src.features.tennis_features import TennisFeatureSnapshot, build_tennis_feature_snapshot
from src.model.rebuild_spec import ReplacementModelSpec
from src.model.tennis_model import TennisProbabilityDecomposition, predict_tennis_match
from src.repositories.snapshot_repository import SnapshotRepository


@dataclass(frozen=True)
class TennisPredictionArtifacts:
    snapshot: TennisMatchSnapshot
    feature_snapshot: TennisFeatureSnapshot
    decomposition: TennisProbabilityDecomposition
    summary: str
    recommendation: str
    model_variant: str = "tennis_match_v1"


def build_tennis_match_snapshot(match: dict, match_data: dict) -> TennisMatchSnapshot:
    return TennisMatchSnapshot.from_match_and_data(match, match_data)


def predict_tennis(match: dict, match_data: dict) -> TennisPredictionArtifacts:
    snapshot = build_tennis_match_snapshot(match, match_data)
    features = build_tennis_feature_snapshot(snapshot, match_data)
    decomposition = predict_tennis_match(features)
    recommendation = classify_tennis_recommendation(decomposition, features)
    summary = summarize_tennis_prediction(snapshot, features, decomposition, recommendation)
    return TennisPredictionArtifacts(
        snapshot=snapshot,
        feature_snapshot=features,
        decomposition=decomposition,
        summary=summary,
        recommendation=recommendation,
    )


def classify_tennis_recommendation(
    decomposition: TennisProbabilityDecomposition,
    features: TennisFeatureSnapshot,
) -> str:
    model_side_prob = max(decomposition.final_probability, 1.0 - decomposition.final_probability)
    market_price = (
        features.market_player_a_price
        if decomposition.final_probability >= 0.5
        else features.market_player_b_price
    )
    if market_price is None:
        return "MONITOR" if model_side_prob < 0.60 else "PASS"
    edge = model_side_prob - float(market_price)
    if model_side_prob >= 0.62 and edge >= 0.04:
        return "BET"
    if model_side_prob >= 0.57 and edge >= 0.02:
        return "PASS"
    return "MONITOR"


def summarize_tennis_prediction(
    snapshot: TennisMatchSnapshot,
    features: TennisFeatureSnapshot,
    decomposition: TennisProbabilityDecomposition,
    recommendation: str,
) -> str:
    favorite = snapshot.player_a if decomposition.final_probability >= 0.5 else snapshot.player_b
    favorite_prob = max(decomposition.final_probability, 1.0 - decomposition.final_probability)
    return (
        f"{snapshot.tour} | {snapshot.player_a} vs {snapshot.player_b} | "
        f"{snapshot.tournament} {snapshot.round_name} | "
        f"{snapshot.surface} | favorite={favorite} {favorite_prob:.1%} | "
        f"recommendation={recommendation}"
    )


def tennis_replacement_spec() -> ReplacementModelSpec:
    from src.model.rebuild_spec import TENNIS_REPLACEMENT_MODEL

    return TENNIS_REPLACEMENT_MODEL


def build_tennis_prediction_record(
    prediction_id: int | None,
    artifacts: TennisPredictionArtifacts,
) -> PredictionRecord:
    favorite_is_player_a = artifacts.decomposition.final_probability >= 0.5
    market_price = (
        artifacts.feature_snapshot.market_player_a_price
        if favorite_is_player_a
        else artifacts.feature_snapshot.market_player_b_price
    )
    favorite_probability = max(
        artifacts.decomposition.final_probability,
        1.0 - artifacts.decomposition.final_probability,
    )
    effective_edge = (
        favorite_probability - float(market_price)
        if market_price is not None
        else None
    )
    return PredictionRecord(
        prediction_id=prediction_id,
        game_id=None,
        sport="TENNIS",
        home_team=artifacts.snapshot.player_a,
        away_team=artifacts.snapshot.player_b,
        game_time=artifacts.snapshot.scheduled_time,
        triage_level=artifacts.snapshot.triage_level,
        base_probability=artifacts.decomposition.base_probability,
        situational_adjustment=artifacts.decomposition.surface_adjustment + artifacts.decomposition.fatigue_adjustment,
        information_edge=artifacts.decomposition.information_adjustment,
        final_probability=artifacts.decomposition.final_probability,
        edge_type="TENNIS",
        recommendation=artifacts.recommendation,
        effective_edge=effective_edge,
        bet_side="home" if favorite_is_player_a else "away",
        model_family="tennis_rule_model",
        model_variant=artifacts.model_variant,
        created_at=datetime.now(timezone.utc).isoformat(),
        decomposition={
            "base": round(artifacts.decomposition.base_probability, 4),
            "surface": round(artifacts.decomposition.surface_adjustment, 4),
            "fatigue": round(artifacts.decomposition.fatigue_adjustment, 4),
            "information": round(artifacts.decomposition.information_adjustment, 4),
            "final": round(artifacts.decomposition.final_probability, 4),
            "tour": artifacts.decomposition.tour,
            "reasoning": list(artifacts.decomposition.reasoning),
        },
        feature_snapshot=artifacts.feature_snapshot.to_dict(),
        snapshot_metadata={
            "collected_at": artifacts.snapshot.collected_at,
            "match_id": artifacts.snapshot.match_id,
            "player_a_id": artifacts.snapshot.player_a_id,
            "player_b_id": artifacts.snapshot.player_b_id,
            "tour": artifacts.snapshot.tour,
            "tournament": artifacts.snapshot.tournament,
            "round_name": artifacts.snapshot.round_name,
        },
    )


def persist_tennis_prediction(
    artifacts: TennisPredictionArtifacts,
    repository: SnapshotRepository | None = None,
) -> PredictionRecord:
    repo = repository or SnapshotRepository()
    snapshot_payload = {
        "snapshot": {
            "game_id": None,
            "match_id": artifacts.snapshot.match_id,
            "sport": "TENNIS",
            "home_team": artifacts.snapshot.player_a,
            "away_team": artifacts.snapshot.player_b,
            "game_time": artifacts.snapshot.scheduled_time,
            "triage_level": artifacts.snapshot.triage_level,
            "collected_at": artifacts.snapshot.collected_at,
            "tour": artifacts.snapshot.tour,
            "tournament": artifacts.snapshot.tournament,
            "round_name": artifacts.snapshot.round_name,
        },
        "feature_snapshot": artifacts.feature_snapshot.to_dict(),
        "model_variant": artifacts.model_variant,
    }
    repo.append_snapshot(snapshot_payload)
    record = build_tennis_prediction_record(None, artifacts)
    repo.append_prediction_record(record)
    return record
