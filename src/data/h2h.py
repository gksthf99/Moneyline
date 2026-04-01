"""
Head-to-head (H2H) game history between two teams.

Fetches recent matchups from ESPN (NBA) and NHL API, returns
season series record and last N meeting results.
"""

import logging
from datetime import date

from src.data import cache
from src.data.nba_espn import get_team_id_by_name, _get as espn_get, BASE_URL as ESPN_BASE
from src.data.nhl_api import _get as nhl_get

logger = logging.getLogger(__name__)


def get_nba_h2h(home_team: str, away_team: str, season: str = "2026") -> dict:
    """Get NBA H2H record between two teams this season.

    Uses ESPN team schedule endpoint to find games between the two teams.

    Returns: {
        "home_wins": int,
        "away_wins": int,
        "total_games": int,
        "home_h2h_pct": float,  # home team's win% in H2H
        "meetings": [{"date": str, "winner": str, "score": str}, ...]
    }
    """
    cache_key = f"nba_h2h_{home_team}_{away_team}_{season}"
    cached = cache.get("historical_h2h", cache_key)
    if cached:
        return cached

    home_id = get_team_id_by_name(home_team)
    if not home_id:
        return _empty_h2h()

    try:
        data = espn_get(f"{ESPN_BASE}/teams/{home_id}/schedule", params={
            "season": season, "seasontype": "2",
        })
    except Exception as e:
        logger.warning("Failed to fetch NBA schedule for %s: %s", home_team, e)
        return _empty_h2h()

    events = data.get("events", [])
    home_wins = 0
    away_wins = 0
    meetings = []

    away_lower = away_team.lower()
    away_last = away_lower.split()[-1] if away_lower else ""

    for e in events:
        comp = e.get("competitions", [{}])[0]
        status = comp.get("status", {}).get("type", {})
        if not status.get("completed", False):
            continue

        competitors = comp.get("competitors", [])
        if len(competitors) != 2:
            continue

        # Check if this game involves the away team
        is_matchup = False
        for t in competitors:
            team_name = t.get("team", {}).get("displayName", "").lower()
            team_last = team_name.split()[-1] if team_name else ""
            if away_last and (away_last == team_last or away_lower in team_name):
                is_matchup = True
                break

        if not is_matchup:
            continue

        # Determine winner
        for t in competitors:
            team_name = t.get("team", {}).get("displayName", "")
            if t.get("winner", False):
                winner = team_name
                break
        else:
            winner = "?"

        # Did home team win this meeting?
        home_name_lower = home_team.lower()
        winner_lower = winner.lower()
        if home_name_lower.split()[-1] == winner_lower.split()[-1]:
            home_wins += 1
        else:
            away_wins += 1

        score_parts = []
        for t in sorted(competitors, key=lambda x: x.get("homeAway", "")):
            s = t.get("score", "?")
            if isinstance(s, dict):
                s = s.get("displayValue", s.get("value", "?"))
            score_parts.append(str(int(float(s))) if s != "?" else "?")
        score = "-".join(score_parts)

        meetings.append({
            "date": e.get("date", "")[:10],
            "winner": winner.split()[-1],
            "score": score,
        })

    total = home_wins + away_wins
    result = {
        "home_wins": home_wins,
        "away_wins": away_wins,
        "total_games": total,
        "home_h2h_pct": home_wins / total if total > 0 else 0.5,
        "meetings": meetings[-5:],  # last 5 meetings
    }

    if total > 0:
        cache.put("historical_h2h", result, cache_key)

    return result


def get_nhl_h2h(home_team: str, away_team: str, home_abbrev: str = "", away_abbrev: str = "") -> dict:
    """Get NHL H2H record between two teams this season.

    Uses NHL club-schedule-season endpoint to find matchups.

    Returns same format as get_nba_h2h.
    """
    if not home_abbrev:
        home_abbrev = _nhl_abbrev(home_team)
    if not home_abbrev:
        return _empty_h2h()

    cache_key = f"nhl_h2h_{home_abbrev}_{away_abbrev or away_team}"
    cached = cache.get("historical_h2h", cache_key)
    if cached:
        return cached

    try:
        data = nhl_get(f"/club-schedule-season/{home_abbrev}/now")
    except Exception as e:
        logger.warning("Failed to fetch NHL schedule for %s: %s", home_abbrev, e)
        return _empty_h2h()

    games = data.get("games", [])
    home_wins = 0
    away_wins = 0
    meetings = []

    for g in games:
        game_state = g.get("gameState", "")
        if game_state not in ("OFF", "FINAL"):
            continue

        # homeTeam/awayTeam are either dicts or strings
        game_home = g.get("homeTeam", {})
        game_away = g.get("awayTeam", {})
        game_home_abbrev = game_home.get("abbrev", game_home) if isinstance(game_home, dict) else str(game_home)
        game_away_abbrev = game_away.get("abbrev", game_away) if isinstance(game_away, dict) else str(game_away)

        # Check if this game involves the opponent
        is_match = False
        if away_abbrev:
            if game_home_abbrev == away_abbrev or game_away_abbrev == away_abbrev:
                is_match = True
        if not is_match:
            continue

        # Determine scores
        home_score = game_home.get("score", 0) if isinstance(game_home, dict) else 0
        away_score = game_away.get("score", 0) if isinstance(game_away, dict) else 0

        # Did our team (home_abbrev) win this game?
        if game_home_abbrev == home_abbrev:
            our_score = home_score
            their_score = away_score
        else:
            our_score = away_score
            their_score = home_score

        if our_score > their_score:
            home_wins += 1
            winner = home_team.split()[-1]
        else:
            away_wins += 1
            winner = away_team.split()[-1]

        meetings.append({
            "date": g.get("gameDate", "")[:10],
            "winner": winner,
            "score": f"{home_score}-{away_score}",
        })

    total = home_wins + away_wins
    result = {
        "home_wins": home_wins,
        "away_wins": away_wins,
        "total_games": total,
        "home_h2h_pct": home_wins / total if total > 0 else 0.5,
        "meetings": meetings[-5:],
    }

    if total > 0:
        cache.put("historical_h2h", result, cache_key)

    return result


# NHL team name → abbreviation map
_NHL_ABBREVS = {
    "anaheim ducks": "ANA", "arizona coyotes": "ARI", "boston bruins": "BOS",
    "buffalo sabres": "BUF", "calgary flames": "CGY", "carolina hurricanes": "CAR",
    "chicago blackhawks": "CHI", "colorado avalanche": "COL", "columbus blue jackets": "CBJ",
    "dallas stars": "DAL", "detroit red wings": "DET", "edmonton oilers": "EDM",
    "florida panthers": "FLA", "los angeles kings": "LAK", "minnesota wild": "MIN",
    "montreal canadiens": "MTL", "montréal canadiens": "MTL",
    "nashville predators": "NSH", "new jersey devils": "NJD",
    "new york islanders": "NYI", "new york rangers": "NYR",
    "ottawa senators": "OTT", "philadelphia flyers": "PHI",
    "pittsburgh penguins": "PIT", "san jose sharks": "SJS",
    "seattle kraken": "SEA", "st. louis blues": "STL",
    "tampa bay lightning": "TBL", "toronto maple leafs": "TOR",
    "utah hockey club": "UTA", "utah mammoth": "UTA",
    "vancouver canucks": "VAN", "vegas golden knights": "VGK",
    "washington capitals": "WSH", "winnipeg jets": "WPG",
}


def _nhl_abbrev(team_name: str) -> str:
    return _NHL_ABBREVS.get(team_name.lower().strip(), "")


def _empty_h2h() -> dict:
    return {
        "home_wins": 0, "away_wins": 0, "total_games": 0,
        "home_h2h_pct": 0.5, "meetings": [],
    }
