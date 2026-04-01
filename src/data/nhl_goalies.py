"""
NHL goalie stats and starter/backup detection.

Fetches per-team goalie stats from NHL API, determines starter vs backup
by games started, and computes save% differential for probability adjustment.
"""

import logging
from datetime import date

from src.data import cache
from src.data.nhl_api import _get

logger = logging.getLogger(__name__)

# League average save% — used as baseline for adjustment
LEAGUE_AVG_SV_PCT = 0.905


def get_team_goalies(team_abbrev: str) -> list[dict]:
    """Get goalie stats for a team this season.

    Returns list of goalies sorted by games started (desc), each with:
        name, games_played, games_started, save_pct, gaa, wins, losses,
        is_starter (True for goalie with most starts)
    """
    cache_key = f"nhl_goalie_stats_{team_abbrev}"
    cached = cache.get("team_season_stats", cache_key)
    if cached:
        return cached

    try:
        data = _get(f"/club-stats/{team_abbrev}/now")
    except Exception as e:
        logger.warning("Failed to fetch goalie stats for %s: %s", team_abbrev, e)
        return []

    goalies_raw = data.get("goalies", [])
    goalies = []

    for g in goalies_raw:
        fn = g.get("firstName", {})
        ln = g.get("lastName", {})
        name = f"{fn.get('default', '')} {ln.get('default', '')}".strip()

        goalies.append({
            "name": name,
            "player_id": g.get("playerId", 0),
            "games_played": g.get("gamesPlayed", 0),
            "games_started": g.get("gamesStarted", 0),
            "save_pct": g.get("savePercentage", 0),
            "gaa": g.get("goalsAgainstAverage", 0),
            "wins": g.get("wins", 0),
            "losses": g.get("losses", 0),
            "ot_losses": g.get("overtimeLosses", 0),
            "shutouts": g.get("shutouts", 0),
        })

    # Sort by games started descending — first is starter
    goalies.sort(key=lambda x: x["games_started"], reverse=True)

    if goalies:
        goalies[0]["is_starter"] = True
        for g in goalies[1:]:
            g["is_starter"] = False

    cache.put("team_season_stats", goalies, cache_key)
    return goalies


def identify_goalie(confirmed_name: str | None, team_abbrev: str) -> dict | None:
    """Match a confirmed goalie name to their stats.

    Args:
        confirmed_name: Goalie name from Daily Faceoff (e.g., "Connor Hellebuyck")
        team_abbrev: 3-letter team code

    Returns goalie stat dict or None.
    """
    if not confirmed_name:
        return None

    goalies = get_team_goalies(team_abbrev)
    if not goalies:
        return None

    confirmed_lower = confirmed_name.lower().strip()
    confirmed_last = confirmed_lower.split()[-1] if confirmed_lower else ""

    for g in goalies:
        g_lower = g["name"].lower()
        g_last = g_lower.split()[-1] if g_lower else ""
        if confirmed_last == g_last or confirmed_lower == g_lower:
            return g

    # Fuzzy: check if last name is contained
    for g in goalies:
        if confirmed_last in g["name"].lower():
            return g

    return None


def goalie_adjustment(
    home_goalie_name: str | None,
    away_goalie_name: str | None,
    home_abbrev: str,
    away_abbrev: str,
) -> tuple[float, str]:
    """Calculate probability adjustment based on goalie matchup.

    Compares confirmed starter's save% to team's #1 starter save%.
    If a backup is playing, adjusts probability toward the opponent.

    Returns (adjustment, description) from home team's perspective.
    Positive = home advantage, Negative = away advantage.
    Capped at ±4%.
    """
    home_goalies = get_team_goalies(home_abbrev)
    away_goalies = get_team_goalies(away_abbrev)

    if not home_goalies or not away_goalies:
        return 0.0, ""

    home_starter = home_goalies[0]  # team's #1 by games started
    away_starter = away_goalies[0]

    # Identify confirmed goalies
    home_confirmed = identify_goalie(home_goalie_name, home_abbrev) if home_goalie_name else None
    away_confirmed = identify_goalie(away_goalie_name, away_abbrev) if away_goalie_name else None

    # Calculate save% differential from expected (team's #1 starter)
    home_delta = 0.0
    away_delta = 0.0
    reasons = []

    if home_confirmed:
        home_delta = home_confirmed["save_pct"] - home_starter["save_pct"]
        if not home_confirmed["is_starter"]:
            reasons.append(
                f"Home backup {home_confirmed['name']} "
                f"(SV% {home_confirmed['save_pct']:.3f} vs starter {home_starter['name']} {home_starter['save_pct']:.3f})"
            )
        elif abs(home_delta) > 0.005:
            reasons.append(f"Home starter {home_confirmed['name']} SV% {home_confirmed['save_pct']:.3f}")

    if away_confirmed:
        away_delta = away_confirmed["save_pct"] - away_starter["save_pct"]
        if not away_confirmed["is_starter"]:
            reasons.append(
                f"Away backup {away_confirmed['name']} "
                f"(SV% {away_confirmed['save_pct']:.3f} vs starter {away_starter['name']} {away_starter['save_pct']:.3f})"
            )
        elif abs(away_delta) > 0.005:
            reasons.append(f"Away starter {away_confirmed['name']} SV% {away_confirmed['save_pct']:.3f}")

    # Net adjustment: better home goalie = positive, better away goalie = negative
    # Scale: 1% save% difference ≈ 4% win probability shift (empirical)
    SAVE_PCT_TO_WIN_PCT = 4.0
    raw_adj = (home_delta - away_delta) * SAVE_PCT_TO_WIN_PCT

    # Also: if one team has a backup and the other has a starter, additional penalty
    home_is_backup = home_confirmed and not home_confirmed["is_starter"]
    away_is_backup = away_confirmed and not away_confirmed["is_starter"]

    if home_is_backup and not away_is_backup:
        raw_adj -= 0.02  # 2% extra penalty for home backup
        reasons.append("Home dressing backup vs away starter")
    elif away_is_backup and not home_is_backup:
        raw_adj += 0.02  # 2% bonus for away backup
        reasons.append("Away dressing backup vs home starter")

    # Cap at ±4%
    adj = max(-0.04, min(0.04, raw_adj))

    if abs(adj) < 0.005:
        return 0.0, ""

    desc = f"Goalie: {'; '.join(reasons)}" if reasons else ""
    return adj, desc
