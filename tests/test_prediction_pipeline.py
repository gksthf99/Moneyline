from dataclasses import dataclass, field

from src.features.game_features import FeatureSnapshot
from src.pipelines.prediction_pipeline import PredictionPipeline


@dataclass
class _InfoEdge:
    total: float = 0.0


@dataclass
class _Decomposition:
    edge: object | None = None
    away_edge: object | None = None
    bet_side: str | None = None
    information_edge: _InfoEdge = field(default_factory=_InfoEdge)


def test_pipeline_collects_snapshot_and_predicts_monitor():
    game = {
        "id": 1,
        "sport": "NBA",
        "home_team": "Bulls",
        "away_team": "Knicks",
        "game_time": "2026-03-29T23:00:00Z",
        "triage_level": "standard",
    }

    def collector(raw_game: dict) -> dict:
        return {"sport": raw_game["sport"], "collected": True}

    def decompose(raw_game: dict, data: dict) -> _Decomposition:
        assert data["collected"] is True
        return _Decomposition()

    def summarize(decomposition: _Decomposition, data: dict) -> str:
        return "summary"

    pipeline = PredictionPipeline(
        collector=collector,
        feature_builder=lambda game, data, decomposition: FeatureSnapshot(
            sport=game["sport"], home_team=game["home_team"], away_team=game["away_team"]
        ),
        decompose=decompose,
        summarize=summarize,
    )
    snapshot = pipeline.collect_snapshot(game)
    artifacts = pipeline.predict(snapshot)

    assert snapshot.home_team == "Bulls"
    assert artifacts.summary == "summary"
    assert artifacts.feature_snapshot.sport == "NBA"
    assert artifacts.edge_type == "B"
    assert artifacts.recommendation == "MONITOR"


def test_pipeline_classifies_information_edge_and_bet():
    def collector(_: dict) -> dict:
        return {}

    def decompose(_: dict, __: dict) -> _Decomposition:
        return _Decomposition(edge=object(), bet_side="away", information_edge=_InfoEdge(total=0.02))

    pipeline = PredictionPipeline(
        collector=collector,
        feature_builder=lambda game, data, decomposition: FeatureSnapshot(
            sport=game["sport"], home_team=game["home_team"], away_team=game["away_team"]
        ),
        decompose=decompose,
        summarize=lambda *_: "ok",
    )
    artifacts = pipeline.predict(pipeline.collect_snapshot({
        "sport": "NHL",
        "home_team": "Stars",
        "away_team": "Avalanche",
        "game_time": "2026-03-29T23:00:00Z",
    }))

    assert artifacts.edge_type == "C"
    assert artifacts.recommendation == "BET"
