"""
NBA.com Stats API integration for team-level advanced stats.

Fetches ORtg, DRtg, NetRtg, Pace per 100 possessions for all 30 teams.
Used by R1.1 to replace win%-based Elo with adjusted net rating as Layer 1 base.
"""

import logging
import time
from datetime import date

import requests

from src.data import cache

logger = logging.getLogger(__name__)

# NBA.com stats API requires specific headers
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.nba.com/",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Origin": "https://www.nba.com",
}

_BASE_URL = "https://stats.nba.com/stats"

# Map NBA.com team names to display names used in our system (ESPN names)
_TEAM_NAME_MAP = {
    "Atlanta Hawks": "Atlanta Hawks",
    "Boston Celtics": "Boston Celtics",
    "Brooklyn Nets": "Brooklyn Nets",
    "Charlotte Hornets": "Charlotte Hornets",
    "Chicago Bulls": "Chicago Bulls",
    "Cleveland Cavaliers": "Cleveland Cavaliers",
    "Dallas Mavericks": "Dallas Mavericks",
    "Denver Nuggets": "Denver Nuggets",
    "Detroit Pistons": "Detroit Pistons",
    "Golden State Warriors": "Golden State Warriors",
    "Houston Rockets": "Houston Rockets",
    "Indiana Pacers": "Indiana Pacers",
    "LA Clippers": "LA Clippers",
    "Los Angeles Lakers": "Los Angeles Lakers",
    "Memphis Grizzlies": "Memphis Grizzlies",
    "Miami Heat": "Miami Heat",
    "Milwaukee Bucks": "Milwaukee Bucks",
    "Minnesota Timberwolves": "Minnesota Timberwolves",
    "New Orleans Pelicans": "New Orleans Pelicans",
    "New York Knicks": "New York Knicks",
    "Oklahoma City Thunder": "Oklahoma City Thunder",
    "Orlando Magic": "Orlando Magic",
    "Philadelphia 76ers": "Philadelphia 76ers",
    "Phoenix Suns": "Phoenix Suns",
    "Portland Trail Blazers": "Portland Trail Blazers",
    "Sacramento Kings": "Sacramento Kings",
    "San Antonio Spurs": "San Antonio Spurs",
    "Toronto Raptors": "Toronto Raptors",
    "Utah Jazz": "Utah Jazz",
    "Washington Wizards": "Washington Wizards",
}


def _fetch_team_stats(
    measure_type: str = "Advanced",
    last_n_games: int = 0,
    season: str = "2025-26",
) -> list[dict]:
    """Fetch team stats from NBA.com stats API.

    Args:
        measure_type: "Advanced" for ORtg/DRtg/NetRtg, "Base" for basic stats
        last_n_games: 0 = full season, 10 = last 10 games, etc.
        season: NBA season string (e.g., "2025-26")

    Returns:
        List of team stat dicts with keys matching column headers.
    """
    params = {
        "Conference": "",
        "DateFrom": "",
        "DateTo": "",
        "Division": "",
        "GameScope": "",
        "GameSegment": "",
        "Height": "",
        "LastNGames": last_n_games,
        "LeagueID": "00",
        "Location": "",
        "MeasureType": measure_type,
        "Month": 0,
        "OpponentTeamID": 0,
        "Outcome": "",
        "PORound": 0,
        "PaceAdjust": "N",
        "PerMode": "PerGame",
        "Period": 0,
        "PlayerExperience": "",
        "PlayerPosition": "",
        "PlusMinus": "N",
        "Rank": "N",
        "Season": season,
        "SeasonSegment": "",
        "SeasonType": "Regular Season",
        "ShotClockRange": "",
        "StarterBench": "",
        "TeamID": 0,
        "TwoWay": 0,
        "VsConference": "",
        "VsDivision": "",
    }

    resp = requests.get(
        f"{_BASE_URL}/leaguedashteamstats",
        params=params,
        headers=_HEADERS,
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()

    result_set = data.get("resultSets", [{}])[0]
    headers = result_set.get("headers", [])
    rows = result_set.get("rowSet", [])

    return [dict(zip(headers, row)) for row in rows]


def _parse_ratings(raw_rows: list[dict]) -> dict[str, dict]:
    """Parse NBA.com API response into team ratings dict.

    Returns:
        {
            "Boston Celtics": {
                "ortg": 118.2,
                "drtg": 110.1,
                "net_rtg": 8.1,
                "pace": 99.3,
                "gp": 70,
            },
            ...
        }
    """
    teams = {}
    for row in raw_rows:
        nba_name = row.get("TEAM_NAME", "")
        display_name = _TEAM_NAME_MAP.get(nba_name, nba_name)

        teams[display_name] = {
            "ortg": row.get("OFF_RATING", 0.0),
            "drtg": row.get("DEF_RATING", 0.0),
            "net_rtg": row.get("NET_RATING", 0.0),
            "pace": row.get("PACE", 0.0),
            "gp": row.get("GP", 0),
        }

    return teams


def get_team_ratings(season: str = "2025-26") -> dict[str, dict]:
    """Fetch season-long ORtg, DRtg, NetRtg, Pace for all 30 NBA teams.

    Uses 1-hour cache (team_advanced TTL in config).

    Returns:
        {
            "Boston Celtics": {"ortg": 118.2, "drtg": 110.1, "net_rtg": 8.1, "pace": 99.3, "gp": 70},
            ...
        }
    """
    cached = cache.get("team_advanced", "nba_ratings_season")
    if cached:
        return cached

    try:
        raw = _fetch_team_stats(measure_type="Advanced", last_n_games=0, season=season)
        ratings = _parse_ratings(raw)
        cache.put("team_advanced", ratings, "nba_ratings_season")
        logger.info("Fetched NBA team ratings: %d teams", len(ratings))
        return ratings
    except Exception as e:
        logger.warning("Failed to fetch NBA team ratings: %s", e)
        return {}


def get_team_ratings_recent(n_games: int = 10, season: str = "2025-26") -> dict[str, dict]:
    """Fetch ORtg/DRtg/NetRtg for last N games only.

    Separate cache key from season-long ratings.
    """
    ck = f"nba_ratings_l{n_games}"
    cached = cache.get("team_advanced", ck)
    if cached:
        return cached

    try:
        raw = _fetch_team_stats(measure_type="Advanced", last_n_games=n_games, season=season)
        ratings = _parse_ratings(raw)
        cache.put("team_advanced", ratings, ck)
        logger.info("Fetched NBA L%d ratings: %d teams", n_games, len(ratings))
        return ratings
    except Exception as e:
        logger.warning("Failed to fetch NBA L%d ratings: %s", n_games, e)
        return {}


def get_team_ratings_clutch(season: str = "2025-26") -> dict[str, dict]:
    """Fetch clutch-time NetRtg (final 5 min, within 5 pts).

    Used as a proxy for garbage-time filtering: teams that perform well
    in competitive minutes are more trustworthy than blowout inflaters.
    """
    ck = "nba_ratings_clutch"
    cached = cache.get("team_advanced", ck)
    if cached:
        return cached

    params = {
        "Conference": "",
        "DateFrom": "",
        "DateTo": "",
        "Division": "",
        "GameScope": "",
        "GameSegment": "",
        "Height": "",
        "LastNGames": 0,
        "LeagueID": "00",
        "Location": "",
        "MeasureType": "Advanced",
        "Month": 0,
        "OpponentTeamID": 0,
        "Outcome": "",
        "PORound": 0,
        "PaceAdjust": "N",
        "PerMode": "PerGame",
        "Period": 0,
        "PlayerExperience": "",
        "PlayerPosition": "",
        "PlusMinus": "N",
        "Rank": "N",
        "Season": season,
        "SeasonSegment": "",
        "SeasonType": "Regular Season",
        "ShotClockRange": "",
        "StarterBench": "",
        "TeamID": 0,
        "TwoWay": 0,
        "VsConference": "",
        "VsDivision": "",
        # Clutch filters
        "AheadBehind": "Ahead or Behind",
        "ClutchTime": "Last 5 Minutes",
        "PointDiff": 5,
    }

    try:
        resp = requests.get(
            f"{_BASE_URL}/leaguedashteamclutch",
            params=params,
            headers=_HEADERS,
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()

        result_set = data.get("resultSets", [{}])[0]
        headers_list = result_set.get("headers", [])
        rows = result_set.get("rowSet", [])

        teams = {}
        for row in rows:
            row_dict = dict(zip(headers_list, row))
            nba_name = row_dict.get("TEAM_NAME", "")
            display_name = _TEAM_NAME_MAP.get(nba_name, nba_name)
            teams[display_name] = {
                "net_rtg": row_dict.get("NET_RATING", 0.0),
                "gp": row_dict.get("GP", 0),
            }

        cache.put("team_advanced", teams, ck)
        logger.info("Fetched NBA clutch ratings: %d teams", len(teams))
        return teams
    except Exception as e:
        logger.warning("Failed to fetch NBA clutch ratings: %s", e)
        return {}
