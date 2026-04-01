"""
NBA data from ESPN unofficial API (site.api.espn.com).

Secondary source for: scoreboard, rosters, injuries, game summaries.
No auth required. Unofficial — may change without notice.
"""

import logging

import requests
from datetime import date
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.data import cache

logger = logging.getLogger(__name__)

BASE_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
WEB_BASE_URL = "https://site.web.api.espn.com/apis/site/v2/sports/basketball/nba"


@retry(
    retry=retry_if_exception_type((requests.ConnectionError, requests.Timeout)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=5, min=5, max=30),
    before_sleep=lambda rs: logger.warning("ESPN NBA request failed (%s), retrying in %ds...", rs.outcome.exception(), rs.next_action.sleep),
)
def _get(url: str, params: dict | None = None) -> dict:
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()


def get_scoreboard(game_date: date | None = None) -> list[dict]:
    """Get today's NBA scoreboard (all games with scores/status)."""
    d = game_date or date.today()
    date_str = d.strftime("%Y%m%d")

    cached = cache.get("schedule", "nba_espn", date_str)
    if cached:
        return cached

    data = _get(f"{BASE_URL}/scoreboard", params={"dates": date_str})
    events = data.get("events", [])

    games = []
    for event in events:
        game = {
            "espn_id": event.get("id"),
            "name": event.get("name"),
            "date": event.get("date"),
            "status": event.get("status", {}).get("type", {}).get("description"),
        }
        competitions = event.get("competitions", [])
        if competitions:
            comp = competitions[0]
            for team_data in comp.get("competitors", []):
                role = "home" if team_data.get("homeAway") == "home" else "away"
                game[f"{role}_team"] = team_data.get("team", {}).get("displayName")
                game[f"{role}_team_abbrev"] = team_data.get("team", {}).get("abbreviation")
                game[f"{role}_score"] = team_data.get("score")
        games.append(game)

    cache.put("schedule", games, "nba_espn", date_str)
    return games


def get_team_roster(team_id: str) -> list[dict]:
    """Get team roster. team_id is ESPN's numeric team ID."""
    cached = cache.get("team_season_stats", "nba_espn_roster", team_id)
    if cached:
        return cached

    data = _get(f"{BASE_URL}/teams/{team_id}/roster")
    athletes = data.get("athletes", [])

    roster = []
    for group in athletes:
        for player in group.get("items", []):
            roster.append({
                "id": player.get("id"),
                "name": player.get("fullName"),
                "position": player.get("position", {}).get("abbreviation"),
                "jersey": player.get("jersey"),
            })

    cache.put("team_season_stats", roster, "nba_espn_roster", team_id)
    return roster


def get_injuries(team_id: str | None = None) -> list[dict]:
    """Get injury reports. Optionally filter by team."""
    cache_key_suffix = f"espn_{team_id}" if team_id else "espn_all"
    cached = cache.get("injury_reports", cache_key_suffix)
    if cached:
        return cached

    if team_id:
        data = _get(f"{BASE_URL}/teams/{team_id}/injuries")
    else:
        # Get all teams' injuries via scoreboard enrichment
        data = _get(f"{BASE_URL}/injuries")

    injuries = []

    # ESPN returns injuries nested by team: {"injuries": [{"displayName": "Team", "injuries": [...]}]}
    teams = data.get("injuries", [])
    if isinstance(teams, list) and teams and isinstance(teams[0], dict) and "injuries" in teams[0]:
        for team_obj in teams:
            team_name = team_obj.get("displayName", "")
            for item in team_obj.get("injuries", []):
                injuries.append({
                    "player": item.get("athlete", {}).get("displayName"),
                    "team": team_name,
                    "status": item.get("type", {}).get("description") or item.get("status", ""),
                    "description": item.get("longComment") or item.get("shortComment"),
                })
    else:
        # Fallback: old flat structure
        for item in data.get("items", []):
            injuries.append({
                "player": item.get("athlete", {}).get("displayName"),
                "team": item.get("team", {}).get("displayName"),
                "status": item.get("status"),
                "description": item.get("longComment") or item.get("shortComment"),
            })

    cache.put("injury_reports", injuries, cache_key_suffix)
    return injuries


def get_game_summary(event_id: str) -> dict:
    """Get detailed game summary / box score."""
    data = _get(f"{WEB_BASE_URL}/summary", params={
        "event": event_id,
        "region": "us",
        "lang": "en",
        "contentorigin": "espn",
    })
    return data


def get_team_recent_form(team_id: str, n: int = 10) -> dict:
    """Get last N games W/L record and current streak for a team.

    Returns: {record: "7-3", streak: "W2", games: [...]}
    """
    cached = cache.get("team_season_stats", "nba_espn_form", team_id)
    if cached:
        return cached

    try:
        data = _get(f"{BASE_URL}/teams/{team_id}/schedule", params={
            "season": "2026", "seasontype": "2",
        })
    except Exception:
        return {"record": "?", "streak": "?", "games": []}

    events = data.get("events", [])
    completed = [
        e for e in events
        if e.get("competitions", [{}])[0].get("status", {}).get("type", {}).get("completed", False)
    ]
    last_n = completed[-n:]

    wins = 0
    streak_count = 0
    last_result = None
    games = []

    for e in last_n:
        comp = e["competitions"][0]
        won = False
        opp = ""
        for t in comp.get("competitors", []):
            if t["team"]["id"] == team_id:
                won = t.get("winner", False)
            else:
                opp = t["team"].get("abbreviation", "?")
        result = "W" if won else "L"
        if result == last_result:
            streak_count += 1
        else:
            streak_count = 1
            last_result = result
        if won:
            wins += 1
        games.append({"opp": opp, "result": result})

    streak_str = f"{last_result}{streak_count}" if last_result else "?"
    form = {
        "record": f"{wins}-{n - wins}",
        "streak": streak_str,
        "games": games,
    }
    cache.put("team_season_stats", form, "nba_espn_form", team_id)
    return form


def get_team_leaders_from_scoreboard(team_name: str, game_date: date | None = None) -> dict:
    """Extract team leaders from scoreboard data.

    Returns: {points: {name, value}, rebounds: {name, value}, assists: {name, value}, rating: {name, line}}
    """
    d = game_date or date.today()
    raw = _get(f"{BASE_URL}/scoreboard", params={"dates": d.strftime("%Y%m%d")})

    for event in raw.get("events", []):
        comp = event.get("competitions", [{}])[0]
        for t in comp.get("competitors", []):
            if t.get("team", {}).get("displayName") == team_name:
                leaders = {}
                records = {}
                for rec in t.get("records", []):
                    records[rec.get("type", "")] = rec.get("summary", "")

                for leader in t.get("leaders", []):
                    key = leader.get("name", "").lower()
                    for a in leader.get("leaders", [])[:1]:
                        athlete = a.get("athlete", {})
                        leaders[key] = {
                            "name": athlete.get("displayName", "?"),
                            "value": a.get("displayValue", "?"),
                        }

                return {
                    "leaders": leaders,
                    "home_record": records.get("home", ""),
                    "away_record": records.get("road", ""),
                }

    return {"leaders": {}, "home_record": "", "away_record": ""}


def get_team_id_by_name(team_name: str) -> str | None:
    """Find ESPN team ID by display name."""
    teams = get_teams()
    for t in teams:
        if t["name"] == team_name:
            return t["espn_id"]
    # Fuzzy: check if team name contains the search
    for t in teams:
        if team_name.lower() in t["name"].lower() or t["name"].lower() in team_name.lower():
            return t["espn_id"]
    return None


def get_teams() -> list[dict]:
    """Get all NBA teams with ESPN IDs."""
    cached = cache.get("team_season_stats", "nba_espn_teams")
    if cached:
        return cached

    data = _get(f"{BASE_URL}/teams")
    teams_data = data.get("sports", [{}])[0].get("leagues", [{}])[0].get("teams", [])

    teams = []
    for t in teams_data:
        team = t.get("team", {})
        teams.append({
            "espn_id": team.get("id"),
            "name": team.get("displayName"),
            "abbreviation": team.get("abbreviation"),
            "location": team.get("location"),
        })

    cache.put("team_season_stats", teams, "nba_espn_teams")
    return teams
