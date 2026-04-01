from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass(frozen=True)
class GameSnapshot:
    """Point-in-time snapshot of a game's metadata and collected features."""

    game_id: int | None
    sport: str
    home_team: str
    away_team: str
    game_time: str
    triage_level: str = "standard"
    collected_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    raw_game: dict = field(default_factory=dict)
    data: dict = field(default_factory=dict)

    @classmethod
    def from_game_and_data(cls, game: dict, data: dict) -> "GameSnapshot":
        return cls(
            game_id=game.get("id"),
            sport=(game.get("sport") or "").upper(),
            home_team=game.get("home_team", ""),
            away_team=game.get("away_team", ""),
            game_time=game.get("game_time", ""),
            triage_level=game.get("triage_level", "standard"),
            raw_game=dict(game),
            data=dict(data),
        )

    def to_game_dict(self) -> dict:
        return dict(self.raw_game)
