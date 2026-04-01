from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from src.domain.game_snapshot import GameSnapshot
from src.features.game_features import FeatureSnapshot
from src.model.decomposition import ProbabilityDecomposition


@dataclass(frozen=True)
class PredictionArtifacts:
    snapshot: GameSnapshot
    feature_snapshot: FeatureSnapshot
    decomposition: ProbabilityDecomposition
    summary: str
    edge_type: str
    recommendation: str
    model_variant: str = "base"


class PredictionPipeline:
    """Non-I/O prediction orchestration over an immutable game snapshot."""

    def __init__(
        self,
        collector: Callable[[dict], dict],
        feature_builder: Callable[[dict, dict, ProbabilityDecomposition], FeatureSnapshot],
        decompose: Callable[[dict, dict], ProbabilityDecomposition],
        summarize: Callable[[ProbabilityDecomposition, dict], str],
    ):
        self._collector = collector
        self._feature_builder = feature_builder
        self._decompose = decompose
        self._summarize = summarize

    def collect_snapshot(self, game: dict) -> GameSnapshot:
        data = self._collector(game)
        return GameSnapshot.from_game_and_data(game, data)

    def predict(self, snapshot: GameSnapshot) -> PredictionArtifacts:
        game = snapshot.to_game_dict()
        data = dict(snapshot.data)
        decomposition = self._decompose(game, data)
        feature_snapshot = self._feature_builder(game, data, decomposition)
        summary = self._summarize(decomposition, data)
        edge_type = self._classify_edge_type(decomposition)
        recommendation = self._classify_recommendation(decomposition)
        return PredictionArtifacts(
            snapshot=snapshot,
            feature_snapshot=feature_snapshot,
            decomposition=decomposition,
            summary=summary,
            edge_type=edge_type,
            recommendation=recommendation,
        )

    @staticmethod
    def _classify_edge_type(decomposition: ProbabilityDecomposition) -> str:
        if decomposition.information_edge.total != 0:
            return "C"
        return "B"

    @staticmethod
    def _classify_recommendation(decomposition: ProbabilityDecomposition) -> str:
        has_any_edge = decomposition.edge is not None or decomposition.away_edge is not None
        if decomposition.bet_side is not None:
            return "BET"
        if has_any_edge:
            return "PASS"
        return "MONITOR"
