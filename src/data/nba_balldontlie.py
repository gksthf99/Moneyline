"""
NBA data from balldontlie API (https://api.balldontlie.io).

Primary source for: scores, stats, rosters, injuries.
Requires API key (free tier: 5 req/min).
"""

import requests
from datetime import date

from src.config import BALLDONTLIE_API_KEY
from src.data import cache

BASE_URL = "https://api.balldontlie.io/v1"


def _headers() -> dict:
    return {"Authorization": BALLDONTLIE_API_KEY}


def _get(endpoint: str, params: dict | None = None) -> dict:
    resp = requests.get(f"{BASE_URL}{endpoint}", headers=_headers(), params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()


def get_games(game_date: date | None = None) -> list[dict]:
    """Get games for a specific date (defaults to today)."""
    d = game_date or date.today()
    date_str = d.isoformat()

    cached = cache.get("schedule", "nba", date_str)
    if cached:
        return cached

    data = _get("/games", params={"dates[]": date_str})
    games = data.get("data", [])

    cache.put("schedule", games, "nba", date_str)
    return games


def get_team_season_stats(team_id: int, season: int | None = None) -> dict:
    """Get team season averages."""
    s = season or date.today().year
    cached = cache.get("team_season_stats", "nba", team_id, s)
    if cached:
        return cached

    data = _get(f"/teams/{team_id}/stats", params={"season": s})
    result = data.get("data", {})

    cache.put("team_season_stats", result, "nba", team_id, s)
    return result


def get_player_stats(player_id: int, season: int | None = None) -> list[dict]:
    """Get player game stats for a season."""
    s = season or date.today().year
    cached = cache.get("player_stats", "nba", player_id, s)
    if cached:
        return cached

    data = _get("/stats", params={"player_ids[]": player_id, "seasons[]": s})
    stats = data.get("data", [])

    cache.put("player_stats", stats, "nba", player_id, s)
    return stats


def get_injuries() -> list[dict]:
    """Get current injury reports. Never cached beyond TTL (30min)."""
    cached = cache.get("injury_reports", "nba_bdl")
    if cached:
        return cached

    data = _get("/player_injuries")
    injuries = data.get("data", [])

    cache.put("injury_reports", injuries, "nba_bdl")
    return injuries


def get_teams() -> list[dict]:
    """Get all NBA teams."""
    cached = cache.get("team_season_stats", "nba_teams_list")
    if cached:
        return cached

    data = _get("/teams")
    teams = data.get("data", [])

    cache.put("team_season_stats", teams, "nba_teams_list")
    return teams
