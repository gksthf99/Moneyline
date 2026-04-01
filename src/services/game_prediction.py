from __future__ import annotations

import os
from datetime import datetime, timezone

from src.domain.prediction_record import PredictionRecord
from src.features.game_features import build_feature_snapshot
from src.model.edge import full_edge_calculation
from src.pipelines.prediction_pipeline import PredictionArtifacts, PredictionPipeline


def _collector(game: dict) -> dict:
    from src.agents.research_agent import collect_nba_data, collect_nhl_data

    sport = (game.get("sport") or "").upper()
    if sport == "NBA":
        return collect_nba_data(game.get("home_team", ""), game.get("away_team", ""))
    return collect_nhl_data(game.get("home_team", ""), game.get("away_team", ""))


def _decompose(game: dict, game_data: dict):
    from src.agents.research_agent import build_decomposition

    return build_decomposition(game, game_data)


def _summarize(decomposition, game_data: dict) -> str:
    from src.agents.research_agent import build_summary

    return build_summary(decomposition, game_data)


def build_prediction_pipeline() -> PredictionPipeline:
    return PredictionPipeline(
        collector=_collector,
        feature_builder=build_feature_snapshot,
        decompose=_decompose,
        summarize=_summarize,
    )


def predict_game(game: dict) -> PredictionArtifacts:
    pipeline = build_prediction_pipeline()
    snapshot = pipeline.collect_snapshot(game)
    artifacts = pipeline.predict(snapshot)

    if _ensemble_enabled():
        try:
            return _apply_experimental_ensemble(artifacts)
        except Exception:
            return artifacts
    return artifacts


def build_prediction_record(prediction_id: int | None, artifacts: PredictionArtifacts) -> PredictionRecord:
    return PredictionRecord.from_artifacts(
        prediction_id=prediction_id,
        snapshot=artifacts.snapshot,
        feature_snapshot=artifacts.feature_snapshot,
        decomposition=artifacts.decomposition,
        edge_type=artifacts.edge_type,
        recommendation=artifacts.recommendation,
        model_variant=artifacts.model_variant,
    )


def _ensemble_enabled() -> bool:
    return os.getenv("ENABLE_EXPERIMENTAL_ENSEMBLE", "").lower() in {"1", "true", "yes", "on"}


def _apply_experimental_ensemble(artifacts: PredictionArtifacts) -> PredictionArtifacts:
    from src.model.ensemble import EnsembleFeatures, get_ensemble

    features = EnsembleFeatures(
        net_rating_spread=(
            (artifacts.feature_snapshot.home_net_rtg.get("season", {}) or {}).get("net_rtg", 0.0)
            - (artifacts.feature_snapshot.away_net_rtg.get("season", {}) or {}).get("net_rtg", 0.0)
        ),
        player_impact_adj=artifacts.decomposition.information_edge.total,
        rest_differential=0.0,
        home_court=1.0,
        pace_differential=(
            (artifacts.feature_snapshot.home_net_rtg.get("season", {}) or {}).get("pace", 100.0)
            - (artifacts.feature_snapshot.away_net_rtg.get("season", {}) or {}).get("pace", 100.0)
        ),
        market_price=artifacts.feature_snapshot.market_home_price,
    )
    ensemble_prob = get_ensemble().predict(features, artifacts.decomposition.final_probability)
    if abs(ensemble_prob - artifacts.decomposition.final_probability) < 1e-9:
        return artifacts

    decomposition = artifacts.decomposition
    decomposition.final_probability = ensemble_prob
    _refresh_edges(decomposition, artifacts)
    return PredictionArtifacts(
        snapshot=artifacts.snapshot,
        feature_snapshot=artifacts.feature_snapshot,
        decomposition=decomposition,
        summary=artifacts.summary,
        edge_type=artifacts.edge_type,
        recommendation=_classify_recommendation(decomposition),
        model_variant="experimental_ensemble",
    )


def _classify_recommendation(decomposition) -> str:
    has_any_edge = decomposition.edge is not None or decomposition.away_edge is not None
    if decomposition.bet_side is not None:
        return "BET"
    if has_any_edge:
        return "PASS"
    return "MONITOR"


def _refresh_edges(decomposition, artifacts: PredictionArtifacts) -> None:
    market = artifacts.snapshot.data.get("market") or {}
    if not market:
        return
    hours_to_game = 6.0
    if artifacts.snapshot.game_time:
        try:
            gt = datetime.fromisoformat(artifacts.snapshot.game_time.replace("Z", "+00:00"))
            hours_to_game = max(0.0, (gt - datetime.now(timezone.utc)).total_seconds() / 3600)
        except (TypeError, ValueError):
            pass

    home_ask = market.get("home_ask") or market.get("home_price")
    home_bid = market.get("home_bid") or market.get("home_price")
    away_ask = market.get("away_ask") or market.get("away_price")
    away_bid = market.get("away_bid") or market.get("away_price")

    if home_ask is not None and home_bid is not None:
        decomposition.edge = full_edge_calculation(
            your_probability=decomposition.final_probability,
            ask_price=home_ask,
            bid_price=home_bid,
            hours_to_game=hours_to_game,
            sport=artifacts.snapshot.sport,
        )
    if away_ask is not None and away_bid is not None:
        decomposition.away_edge = full_edge_calculation(
            your_probability=1 - decomposition.final_probability,
            ask_price=away_ask,
            bid_price=away_bid,
            hours_to_game=hours_to_game,
            sport=artifacts.snapshot.sport,
        )
    if decomposition.away_edge is not None:
        decomposition.bet_side = "home" if decomposition.final_probability >= 0.5 else "away"
