from __future__ import annotations

import json
import logging
from dataclasses import asdict, is_dataclass
from pathlib import Path

from src.repositories import supabase

logger = logging.getLogger(__name__)

LOG_DIR = Path(__file__).resolve().parent.parent.parent / "logs"
SNAPSHOT_LOG = LOG_DIR / "prediction_snapshots.jsonl"
RECORD_LOG = LOG_DIR / "prediction_records.jsonl"


class SnapshotRepository:
    """Immutable local persistence for snapshots and prediction records."""

    def __init__(self, snapshot_path: Path = SNAPSHOT_LOG, record_path: Path = RECORD_LOG):
        self.snapshot_path = snapshot_path
        self.record_path = record_path
        self.snapshot_path.parent.mkdir(parents=True, exist_ok=True)

    def append_snapshot(self, payload: object) -> None:
        data = self._normalize(payload)
        self._append_jsonl(self.snapshot_path, data)
        self._persist_snapshot_supabase(data)

    def append_prediction_record(self, payload: object) -> None:
        data = self._normalize(payload)
        self._append_jsonl(self.record_path, data)
        self._persist_record_supabase(data)

    def _append_jsonl(self, path: Path, payload: object) -> None:
        try:
            data = self._normalize(payload)
            with open(path, "a") as handle:
                handle.write(json.dumps(data, default=str))
                handle.write("\n")
        except Exception as exc:
            logger.warning("Snapshot persistence failed [%s]: %s", path.name, exc)

    def _normalize(self, payload: object) -> dict:
        if is_dataclass(payload):
            return asdict(payload)
        return payload

    def _persist_snapshot_supabase(self, data: dict) -> None:
        if not supabase.is_configured():
            return
        row = {
            "game_id": data.get("snapshot", {}).get("game_id"),
            "sport": data.get("snapshot", {}).get("sport"),
            "home_team": data.get("snapshot", {}).get("home_team"),
            "away_team": data.get("snapshot", {}).get("away_team"),
            "game_time": data.get("snapshot", {}).get("game_time"),
            "triage_level": data.get("snapshot", {}).get("triage_level"),
            "collected_at": data.get("snapshot", {}).get("collected_at"),
            "snapshot": data.get("snapshot", {}),
            "feature_snapshot": data.get("feature_snapshot", {}),
            "model_variant": data.get("model_variant", "base"),
        }
        supabase.insert("prediction_snapshots", row)

    def _persist_record_supabase(self, data: dict) -> None:
        if not supabase.is_configured():
            return
        row = {
            "game_id": data.get("game_id"),
            "sport": data.get("sport"),
            "home_team": data.get("home_team"),
            "away_team": data.get("away_team"),
            "game_time": data.get("game_time"),
            "triage_level": data.get("triage_level"),
            "base_probability": data.get("base_probability"),
            "situational_adjustment": data.get("situational_adjustment"),
            "information_edge": data.get("information_edge"),
            "final_probability": data.get("final_probability"),
            "edge_type": data.get("edge_type"),
            "recommendation": data.get("recommendation"),
            "effective_edge": data.get("effective_edge"),
            "bet_side": data.get("bet_side"),
            "model_family": data.get("model_family", "layered_rule_model"),
            "model_variant": data.get("model_variant", "base"),
            "decomposition": data.get("decomposition", {}),
            "feature_snapshot": data.get("feature_snapshot", {}),
            "snapshot_metadata": data.get("snapshot_metadata", {}),
            "created_at": data.get("created_at"),
        }
        supabase.insert("prediction_records", row)
