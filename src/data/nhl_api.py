"""
NHL data from official API (api-web.nhle.com).

Primary source for: schedule, scores, team stats, player stats, standings.
No auth required.
"""

import logging

import requests
from datetime import date
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.data import cache

logger = logging.getLogger(__name__)

BASE_URL = "https://api-web.nhle.com/v1"


@retry(
    retry=retry_if_exception_type((requests.ConnectionError, requests.Timeout)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=5, min=5, max=30),
    before_sleep=lambda rs: logger.warning("NHL API request failed (%s), retrying in %ds...", rs.outcome.exception(), rs.next_action.sleep),
)
def _get(endpoint: str) -> dict | list:
    resp = requests.get(f"{BASE_URL}{endpoint}", timeout=15)
    resp.raise_for_status()
    return resp.json()


def get_schedule(game_date: date | None = None) -> dict:
    """Get NHL schedule for a date (defaults to today)."""
    d = game_date or date.today()
    date_str = d.isoformat()

    cached = cache.get("schedule", "nhl", date_str)
    if cached:
        return cached

    data = _get(f"/schedule/{date_str}")

    cache.put("schedule", data, "nhl", date_str)
    return data


def get_daily_scores(game_date: date | None = None) -> dict:
    """Get scores for a date."""
    d = game_date or date.today()
    date_str = d.isoformat()

    cached = cache.get("schedule", "nhl_scores", date_str)
    if cached:
        return cached

    data = _get(f"/score/{date_str}")

    cache.put("schedule", data, "nhl_scores", date_str)
    return data


def get_standings(game_date: date | None = None) -> dict:
    """Get NHL standings."""
    if game_date:
        date_str = game_date.isoformat()
    else:
        date_str = "now"

    cached = cache.get("standings", "nhl", date_str)
    if cached:
        return cached

    data = _get(f"/standings/{date_str}")

    cache.put("standings", data, "nhl", date_str)
    return data


def get_team_stats(team_abbrev: str, season: str | None = None) -> dict:
    """Get team stats. team_abbrev is 3-letter code (e.g., 'COL', 'TOR').
    season format: '20252026'."""
    if season:
        endpoint = f"/club-stats/{team_abbrev}/{season}/2"
    else:
        endpoint = f"/club-stats/{team_abbrev}/now"

    cached = cache.get("team_season_stats", "nhl", team_abbrev, season or "now")
    if cached:
        return cached

    data = _get(endpoint)

    cache.put("team_season_stats", data, "nhl", team_abbrev, season or "now")
    return data


def get_team_schedule(team_abbrev: str, game_date: date | None = None) -> dict:
    """Get team-specific schedule for the week."""
    d = game_date or date.today()
    date_str = d.isoformat()

    data = _get(f"/club-schedule/{team_abbrev}/week/{date_str}")
    return data


def get_boxscore(game_id: int) -> dict:
    """Get game box score."""
    data = _get(f"/gamecenter/{game_id}/boxscore")
    return data


def get_player_game_log(player_id: int, season: str | None = None) -> dict:
    """Get player's game-by-game stats.
    season format: '20252026'."""
    if season:
        endpoint = f"/player/{player_id}/game-log/{season}/2"
    else:
        endpoint = f"/player/{player_id}/game-log/now"

    cached = cache.get("player_stats", "nhl", player_id, season or "now")
    if cached:
        return cached

    data = _get(endpoint)

    cache.put("player_stats", data, "nhl", player_id, season or "now")
    return data


def get_skater_leaders(season: str | None = None) -> dict:
    """Get top skater stats leaders."""
    if season:
        data = _get(f"/skater-stats-leaders/{season}/2")
    else:
        data = _get("/skater-stats-leaders/current")
    return data


def get_goalie_leaders(season: str | None = None) -> dict:
    """Get top goalie stats leaders."""
    if season:
        data = _get(f"/goalie-stats-leaders/{season}/2")
    else:
        data = _get("/goalie-stats-leaders/current")
    return data
