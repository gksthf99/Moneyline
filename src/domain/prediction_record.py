from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

from src.domain.game_snapshot import GameSnapshot
from src.features.game_features import FeatureSnapshot


@dataclass(frozen=True)
class PredictionRecord:
    prediction_id: int | None
    game_id: int | None
    sport: str
    home_team: str
    away_team: str
    game_time: str
    triage_level: str
    base_probability: float
    situational_adjustment: float
    information_edge: float
    final_probability: float
    edge_type: str
    recommendation: str
    effective_edge: float | None
    bet_side: str | None
    model_family: str = "layered_rule_model"
    model_variant: str = "base"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    decomposition: dict = field(default_factory=dict)
    feature_snapshot: dict = field(default_factory=dict)
    snapshot_metadata: dict = field(default_factory=dict)

    @classmethod
    def from_artifacts(
        cls,
        prediction_id: int | None,
        snapshot: GameSnapshot,
        feature_snapshot: FeatureSnapshot,
        decomposition,
        edge_type: str,
        recommendation: str,
        model_variant: str = "base",
    ) -> "PredictionRecord":
        active_edge = decomposition.edge
        if decomposition.bet_side == "away" and decomposition.away_edge:
            active_edge = decomposition.away_edge

        return cls(
            prediction_id=prediction_id,
            game_id=snapshot.game_id,
            sport=snapshot.sport,
            home_team=snapshot.home_team,
            away_team=snapshot.away_team,
            game_time=snapshot.game_time,
            triage_level=snapshot.triage_level,
            base_probability=decomposition.base_probability,
            situational_adjustment=decomposition.situational_adjustment.total,
            information_edge=decomposition.information_edge.total,
            final_probability=decomposition.final_probability,
            edge_type=edge_type,
            recommendation=recommendation,
            effective_edge=active_edge.effective_edge if active_edge else None,
            bet_side=decomposition.bet_side,
            model_variant=model_variant,
            decomposition=decomposition.to_supabase_dict(),
            feature_snapshot=asdict(feature_snapshot),
            snapshot_metadata={
                "collected_at": snapshot.collected_at,
                "game_id": snapshot.game_id,
            },
        )

    @classmethod
    def from_supabase_row(cls, row: dict, game: dict | None = None) -> "PredictionRecord | None":
        def _load_jsonish(value):
            if isinstance(value, str):
                try:
                    return json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    return {}
            return value if isinstance(value, dict) else {}

        decomp = row.get("prob_decomposition")
        if isinstance(decomp, str):
            try:
                decomp = json.loads(decomp)
            except (json.JSONDecodeError, TypeError):
                return None
        if not isinstance(decomp, dict):
            return None

        game_info = game or {}
        model_metadata = row.get("data_freshness_status")
        model_variant = "base"
        if isinstance(model_metadata, str) and model_metadata.startswith("model_variant="):
            model_variant = model_metadata.split("=", 1)[1]

        return cls(
            prediction_id=row.get("id"),
            game_id=row.get("game_id"),
            sport=game_info.get("sport", ""),
            home_team=game_info.get("home_team", ""),
            away_team=game_info.get("away_team", ""),
            game_time=game_info.get("game_time", ""),
            triage_level=game_info.get("triage_level", "standard"),
            base_probability=decomp.get("base", 0.5),
            situational_adjustment=decomp.get("situational", 0.0),
            information_edge=decomp.get("information", 0.0),
            final_probability=decomp.get("final", 0.5),
            edge_type=row.get("edge_type", "B"),
            recommendation=row.get("recommendation", "MONITOR"),
            effective_edge=row.get("effective_edge"),
            bet_side=decomp.get("bet_side"),
            model_variant=model_variant,
            created_at=row.get("created_at") or datetime.now(timezone.utc).isoformat(),
            decomposition=decomp,
            feature_snapshot=_load_jsonish(row.get("feature_snapshot")),
            snapshot_metadata=_load_jsonish(row.get("snapshot_metadata")),
        )


@dataclass(frozen=True)
class GradingRecord:
    prediction_id: int | None
    game_id: int | None
    home_team: str
    away_team: str
    predicted_home_prob: float
    home_won: bool
    home_score: int
    away_score: int
    brier_score: float
    edge_type: str
    effective_edge: float | None
    recommendation: str
    base_correct: bool
    sit_helped: bool
    info_helped: bool
    model_family: str = "layered_rule_model"
    model_variant: str = "base"

    def to_dict(self) -> dict:
        return asdict(self)
