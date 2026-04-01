"""
NBA.com player-level advanced stats for measured impact model.

Fetches on-court ORtg/DRtg/NetRtg per player. Combined with team-level
ratings, computes each player's measured impact on team performance.

Replaces the generic T1/T2/T3/T4 tier system with actual on/off differentials.
"""

import logging
from src.data import cache
from src.data.nba_stats import _HEADERS, _BASE_URL

import requests

logger = logging.getLogger(__name__)

# Minimum total minutes to be included in impact calculations.
# Below this, sample is too noisy to be useful.
MIN_TOTAL_MINUTES = 300


def get_all_player_ratings(season: str = "2025-26") -> dict[str, list[dict]]:
    """Fetch on-court advanced stats for all NBA players.

    Returns dict keyed by team abbreviation, each containing list of player dicts:
        {
            "ATL": [
                {"name": "Trae Young", "player_id": 1629027, "gp": 65, "mpg": 35.2,
                 "total_min": 2288, "net_rtg": 5.2, "ortg": 118.5, "drtg": 113.3,
                 "usg_pct": 0.29, "pie": 0.142},
                ...
            ],
            ...
        }
    """
    cached = cache.get("team_advanced", "nba_player_ratings")
    if cached:
        return cached

    try:
        resp = requests.get(
            f"{_BASE_URL}/leaguedashplayerstats",
            params={
                "Conference": "", "DateFrom": "", "DateTo": "", "Division": "",
                "GameScope": "", "GameSegment": "", "Height": "",
                "LastNGames": 0, "LeagueID": "00", "Location": "",
                "MeasureType": "Advanced", "Month": 0, "OpponentTeamID": 0,
                "Outcome": "", "PORound": 0, "PaceAdjust": "N",
                "PerMode": "PerGame", "Period": 0, "PlayerExperience": "",
                "PlayerPosition": "", "PlusMinus": "N", "Rank": "N",
                "Season": season, "SeasonSegment": "", "SeasonType": "Regular Season",
                "ShotClockRange": "", "StarterBench": "", "TeamID": 0,
                "TwoWay": 0, "VsConference": "", "VsDivision": "", "Weight": "",
            },
            headers=_HEADERS,
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()

        rs = data.get("resultSets", [{}])[0]
        headers = rs.get("headers", [])
        rows = rs.get("rowSet", [])

        by_team: dict[str, list[dict]] = {}
        for row in rows:
            raw = dict(zip(headers, row))
            gp = raw.get("GP", 0)
            mpg = raw.get("MIN", 0.0)
            total_min = gp * mpg

            if total_min < MIN_TOTAL_MINUTES:
                continue

            team = raw.get("TEAM_ABBREVIATION", "")
            player = {
                "name": raw.get("PLAYER_NAME", ""),
                "player_id": raw.get("PLAYER_ID", 0),
                "team_abbr": team,
                "gp": gp,
                "mpg": round(mpg, 1),
                "total_min": round(total_min, 0),
                "net_rtg": raw.get("NET_RATING", 0.0),
                "ortg": raw.get("OFF_RATING", 0.0),
                "drtg": raw.get("DEF_RATING", 0.0),
                "usg_pct": raw.get("USG_PCT", 0.0),
                "pie": raw.get("PIE", 0.0),
            }

            by_team.setdefault(team, []).append(player)

        # Sort each team's players by minutes (highest first)
        for team in by_team:
            by_team[team].sort(key=lambda p: p["total_min"], reverse=True)

        cache.put("team_advanced", by_team, "nba_player_ratings")
        total = sum(len(v) for v in by_team.values())
        logger.info("Fetched NBA player ratings: %d players across %d teams", total, len(by_team))
        return by_team

    except Exception as e:
        logger.warning("Failed to fetch NBA player ratings: %s", e)
        return {}


def build_player_impact_map(
    team_ratings: dict[str, dict] | None = None,
) -> dict[str, float]:
    """Build a lookup: player_name → measured win probability impact.

    Impact = player's on-court NetRtg minus team's overall NetRtg,
    converted to win probability space via our logistic model.

    A player with on-court NetRtg of +12 on a team with overall +8
    has a raw impact of +4 net rating points. If they're out, the team
    is expected to play at roughly their off-court level (team_avg - impact).

    Returns: {"Shai Gilgeous-Alexander": 0.082, "Jayson Tatum": 0.065, ...}
    Positive = their absence hurts the team (most players).
    """
    from src.data.nba_stats import get_team_ratings

    if team_ratings is None:
        team_ratings = get_team_ratings()

    player_ratings = get_all_player_ratings()
    if not player_ratings or not team_ratings:
        return {}

    # Build team abbrev → team name mapping from team_ratings
    # We need to map NBA.com abbreviations to our display names
    _ABBREV_MAP = {
        "ATL": "Atlanta Hawks", "BOS": "Boston Celtics", "BKN": "Brooklyn Nets",
        "CHA": "Charlotte Hornets", "CHI": "Chicago Bulls", "CLE": "Cleveland Cavaliers",
        "DAL": "Dallas Mavericks", "DEN": "Denver Nuggets", "DET": "Detroit Pistons",
        "GSW": "Golden State Warriors", "HOU": "Houston Rockets", "IND": "Indiana Pacers",
        "LAC": "LA Clippers", "LAL": "Los Angeles Lakers", "MEM": "Memphis Grizzlies",
        "MIA": "Miami Heat", "MIL": "Milwaukee Bucks", "MIN": "Minnesota Timberwolves",
        "NOP": "New Orleans Pelicans", "NYK": "New York Knicks",
        "OKC": "Oklahoma City Thunder", "ORL": "Orlando Magic", "PHI": "Philadelphia 76ers",
        "PHX": "Phoenix Suns", "POR": "Portland Trail Blazers", "SAC": "Sacramento Kings",
        "SAS": "San Antonio Spurs", "TOR": "Toronto Raptors", "UTA": "Utah Jazz",
        "WAS": "Washington Wizards",
    }

    # Logistic conversion factor: how much 1 point of net rating moves win prob
    # At P=0.5, derivative of logistic is ~0.018 per net rating point (with K=16, shrink=0.65)
    # At extremes it's less. Use average slope = 0.015 as approximation.
    NET_RTG_TO_WIN_PROB = 0.015

    # Minutes weighting: players who play more minutes have more impact
    # Scale by fraction of 40 min (a full game)
    FULL_GAME_MIN = 40.0

    impact_map: dict[str, float] = {}

    for team_abbr, players in player_ratings.items():
        team_name = _ABBREV_MAP.get(team_abbr)
        if not team_name or team_name not in team_ratings:
            continue

        team_net = team_ratings[team_name].get("net_rtg", 0.0)

        for p in players:
            # Raw impact: how much better/worse the team is with this player on court
            raw_impact = p["net_rtg"] - team_net

            # Scale by minutes share (35 mpg player has more impact than 15 mpg)
            min_share = min(p["mpg"] / FULL_GAME_MIN, 1.0)

            # Convert to win probability impact
            impact = raw_impact * NET_RTG_TO_WIN_PROB * min_share

            # Floor: ignore players with < 1.5% impact (noise)
            if abs(impact) < 0.015:
                continue

            impact_map[p["name"]] = round(impact, 4)

    logger.info("Built player impact map: %d players with measurable impact", len(impact_map))
    return impact_map
