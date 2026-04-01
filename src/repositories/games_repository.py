from __future__ import annotations

from datetime import date

from src.repositories import supabase
from src.time_utils import sports_day_bounds


class GamesRepository:
    """Supabase-backed read access for scheduled games."""

    def list_slate(self, target_date: date | None = None) -> list[dict]:
        if not supabase.is_configured():
            return []
        lower, upper = sports_day_bounds(target_date)
        return supabase.select(
            "games",
            {
                "and": f"(game_time.gte.{lower},game_time.lt.{upper})",
                "status": "eq.scheduled",
                "select": "id,sport,home_team,away_team,game_time,triage_level",
                "order": "game_time.asc",
            },
        )
