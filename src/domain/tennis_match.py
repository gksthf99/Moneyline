from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass(frozen=True)
class TennisMatchSnapshot:
    """Point-in-time tennis match snapshot for the sport-specific prediction path."""

    match_id: str | None
    player_a_id: str | None
    player_b_id: str | None
    player_a: str
    player_b: str
    tour: str
    tournament: str
    round_name: str
    surface: str
    best_of: int
    scheduled_time: str
    indoor: bool = False
    triage_level: str = "standard"
    collected_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    raw_match: dict = field(default_factory=dict)
    data: dict = field(default_factory=dict)

    @classmethod
    def from_match_and_data(cls, match: dict, data: dict) -> "TennisMatchSnapshot":
        return cls(
            match_id=match.get("id"),
            player_a_id=match.get("player_a_id"),
            player_b_id=match.get("player_b_id"),
            player_a=match.get("player_a", ""),
            player_b=match.get("player_b", ""),
            tour=(match.get("tour") or "ATP").upper(),
            tournament=match.get("tournament", ""),
            round_name=match.get("round_name", ""),
            surface=(match.get("surface") or "").lower(),
            best_of=int(match.get("best_of", 3) or 3),
            scheduled_time=match.get("scheduled_time", ""),
            indoor=bool(match.get("indoor", False)),
            triage_level=match.get("triage_level", "standard"),
            raw_match=dict(match),
            data=dict(data),
        )

    def to_match_dict(self) -> dict:
        return dict(self.raw_match)

    def to_game_dict(self) -> dict:
        """Compatibility bridge for shared storage that still expects home/away keys."""
        return {
            "id": self.match_id,
            "sport": "TENNIS",
            "player_a_id": self.player_a_id,
            "player_b_id": self.player_b_id,
            "home_team": self.player_a,
            "away_team": self.player_b,
            "tour": self.tour,
            "game_time": self.scheduled_time,
            "triage_level": self.triage_level,
            "tournament": self.tournament,
            "round_name": self.round_name,
            "surface": self.surface,
            "best_of": self.best_of,
            "indoor": self.indoor,
        }
