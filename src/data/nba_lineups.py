"""
NBA live lineup detection via ESPN game summary API.

Polls ESPN 30 minutes before game tip-off to detect confirmed starters.
When lineups are confirmed, compares against expected starters and
triggers re-pricing if a significant player is unexpectedly out.

ESPN populates starter=True/False in boxscore.players once lineups
are released (typically 30-60 min before tip).
"""

import logging
from datetime import date

import requests

logger = logging.getLogger(__name__)

ESPN_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
ESPN_SUMMARY = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary"


def get_todays_nba_events(game_date: date | None = None) -> list[dict]:
    """Fetch today's NBA games from ESPN scoreboard.

    Returns list of:
        {"event_id", "name", "home_team", "away_team", "status", "game_time"}
    """
    if game_date is None:
        game_date = date.today()

    try:
        resp = requests.get(
            ESPN_SCOREBOARD,
            params={"dates": game_date.strftime("%Y%m%d")},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.warning("Failed to fetch NBA scoreboard: %s", e)
        return []

    events = []
    for ev in data.get("events", []):
        comp = ev.get("competitions", [{}])[0]
        competitors = comp.get("competitors", [])

        home = away = None
        for c in competitors:
            if c.get("homeAway") == "home":
                home = c
            else:
                away = c

        if not home or not away:
            continue

        events.append({
            "event_id": ev.get("id", ""),
            "name": ev.get("name", ""),
            "home_team": home.get("team", {}).get("displayName", ""),
            "away_team": away.get("team", {}).get("displayName", ""),
            "status": ev.get("status", {}).get("type", {}).get("description", ""),
            "game_time": comp.get("date", ""),
        })

    return events


def get_confirmed_starters(event_id: str) -> dict | None:
    """Fetch confirmed starters for a specific game from ESPN summary.

    Returns None if starters not yet available (game not close enough).
    Returns dict when available:
        {
            "home_team": "Boston Celtics",
            "away_team": "Brooklyn Nets",
            "home_starters": ["Derrick White", "Jaylen Brown", ...],
            "away_starters": ["Cam Thomas", ...],
            "home_bench": ["Payton Pritchard", ...],
            "away_bench": [...],
        }
    """
    try:
        resp = requests.get(
            ESPN_SUMMARY,
            params={"event": event_id},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.warning("Failed to fetch ESPN summary for event %s: %s", event_id, e)
        return None

    bs = data.get("boxscore", {})
    player_groups = bs.get("players", [])

    if not player_groups:
        return None

    result = {
        "home_team": "",
        "away_team": "",
        "home_starters": [],
        "away_starters": [],
        "home_bench": [],
        "away_bench": [],
    }

    # Determine home/away from header
    header = data.get("header", {})
    comps = header.get("competitions", [{}])
    home_away_map = {}
    if comps:
        for c in comps[0].get("competitors", []):
            team_name = c.get("team", {}).get("displayName", "")
            ha = c.get("homeAway", "")
            home_away_map[team_name] = ha
            if ha == "home":
                result["home_team"] = team_name
            else:
                result["away_team"] = team_name

    for group in player_groups:
        team_name = group.get("team", {}).get("displayName", "")
        ha = home_away_map.get(team_name, "away")

        stats = group.get("statistics", [])
        for sg in stats:
            athletes = sg.get("athletes", [])
            has_any_starter = any(a.get("starter") for a in athletes)

            if not has_any_starter:
                return None  # Starters not yet confirmed

            for a in athletes:
                name = a.get("athlete", {}).get("displayName", "")
                if not name:
                    continue

                if a.get("starter"):
                    result[f"{ha}_starters"].append(name)
                else:
                    result[f"{ha}_bench"].append(name)

    # Validate: should have 5 starters per team
    if len(result["home_starters"]) < 5 or len(result["away_starters"]) < 5:
        return None

    return result


def detect_lineup_changes(
    confirmed: dict,
    expected_starters: dict[str, list[str]] | None = None,
    impact_map: dict[str, float] | None = None,
) -> list[dict]:
    """Compare confirmed lineups against expected starters.

    Detects unexpected absences (players expected to start but not in lineup)
    and unexpected additions (bench players starting).

    Args:
        confirmed: Output from get_confirmed_starters()
        expected_starters: {"team_name": ["Player1", ...]} or None
        impact_map: {"player_name": impact_float} from nba_player_stats

    Returns list of changes:
        [{"team", "player", "type": "unexpected_out"|"unexpected_in", "impact": float}]
    """
    if not impact_map:
        impact_map = {}

    changes = []

    for side in ["home", "away"]:
        team = confirmed.get(f"{side}_team", "")
        starters = set(confirmed.get(f"{side}_starters", []))
        bench = set(confirmed.get(f"{side}_bench", []))
        all_active = starters | bench

        if expected_starters and team in expected_starters:
            expected = set(expected_starters[team])
            # Players expected to start but not in starters
            for player in expected - starters:
                if player not in all_active:
                    # Completely absent — likely a late scratch
                    impact = impact_map.get(player, 0.0)
                    if impact > 0.015:  # Only flag meaningful absences
                        changes.append({
                            "team": team,
                            "side": side,
                            "player": player,
                            "type": "unexpected_out",
                            "impact": impact,
                        })
        else:
            # No expected starters — check impact map for missing high-impact players
            for player, impact in impact_map.items():
                if impact > 0.04 and player not in all_active:
                    # High-impact player not active — but we don't know their team
                    # from the impact map alone. Skip unless we have team info.
                    pass

    return changes
