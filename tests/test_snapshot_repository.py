import json
from pathlib import Path

from src.repositories.snapshot_repository import SnapshotRepository


def test_snapshot_repository_appends_jsonl(tmp_path: Path):
    repo = SnapshotRepository(
        snapshot_path=tmp_path / "snapshots.jsonl",
        record_path=tmp_path / "records.jsonl",
    )

    repo.append_snapshot({"game_id": 1, "sport": "NBA"})
    repo.append_prediction_record({"game_id": 1, "final_probability": 0.61})

    snapshot_lines = (tmp_path / "snapshots.jsonl").read_text().strip().splitlines()
    record_lines = (tmp_path / "records.jsonl").read_text().strip().splitlines()

    assert json.loads(snapshot_lines[0])["game_id"] == 1
    assert json.loads(record_lines[0])["final_probability"] == 0.61
