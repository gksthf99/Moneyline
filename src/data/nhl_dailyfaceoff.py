"""
NHL goalie starts and line combinations from Daily Faceoff (dailyfaceoff.com).

Parses structured JSON from __NEXT_DATA__ (Next.js). No HTML guesswork.
Goalie starts and lineups are NEVER cached per gameplan rules — always live pull.
"""

import json
import logging
import requests
from bs4 import BeautifulSoup
from datetime import date

logger = logging.getLogger(__name__)

BASE_URL = "https://www.dailyfaceoff.com"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
}


def _fetch_soup(url: str) -> BeautifulSoup:
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "lxml")


def _fetch_next_data(url: str) -> dict:
    """Extract __NEXT_DATA__ JSON from a Daily Faceoff page."""
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "lxml")
    script = soup.find("script", id="__NEXT_DATA__")
    if not script or not script.string:
        raise ValueError("No __NEXT_DATA__ found on page")
    return json.loads(script.string)


def get_starting_goalies(game_date: date | None = None) -> list[dict]:
    """Get confirmed/projected goalie starts. NEVER cached.

    Parses __NEXT_DATA__ JSON — each matchup has goalie names, stats, and
    confirmation status structurally bound to the correct team. No guesswork.

    Returns list of matchup dicts with:
    - away_team, home_team
    - away_goalie, home_goalie (name strings, None if unconfirmed)
    - away_goalie_stats, home_goalie_stats (dict with save_pct, gaa, wins, losses, otl, shutouts, overall_score)
    - away_status, home_status ("Confirmed", "Likely", or "Unconfirmed")
    - away_notes, home_notes (analyst writeups)
    - away_news_source, home_news_source (source name)
    - game_time (ISO string)
    """
    if game_date:
        url = f"{BASE_URL}/starting-goalies/{game_date.isoformat()}"
    else:
        url = f"{BASE_URL}/starting-goalies"

    try:
        next_data = _fetch_next_data(url)
    except Exception as e:
        logger.warning("Failed to fetch __NEXT_DATA__ from %s: %s", url, e)
        return []

    data = next_data.get("props", {}).get("pageProps", {}).get("data", [])
    if not data:
        logger.warning("No matchup data found in __NEXT_DATA__")
        return []

    matchups = []
    for game in data:
        def _goalie_stats(side: str) -> dict | None:
            name = game.get(f"{side}GoalieName")
            if not name:
                return None
            sv_pct = game.get(f"{side}GoalieSavePercentage")
            return {
                "name": name,
                "save_pct": float(sv_pct) if sv_pct else None,
                "gaa": float(gaa) if (gaa := game.get(f"{side}GoalieGoalsAgainstAvg")) else None,
                "wins": game.get(f"{side}GoalieWins"),
                "losses": game.get(f"{side}GoalieLosses"),
                "otl": game.get(f"{side}GoalieOvertimeLosses"),
                "shutouts": game.get(f"{side}GoalieShutouts"),
                "overall_score": game.get(f"{side}GoalieOverallScore"),
            }

        def _status(side: str) -> str:
            name = game.get(f"{side}NewsStrengthName")
            return name if name else "Unconfirmed"

        matchup = {
            "away_team": game.get("awayTeamName", ""),
            "home_team": game.get("homeTeamName", ""),
            "away_team_slug": game.get("awayTeamSlug", ""),
            "home_team_slug": game.get("homeTeamSlug", ""),
            "game_time": game.get("dateGmt", ""),
            "away_goalie": game.get("awayGoalieName"),
            "home_goalie": game.get("homeGoalieName"),
            "away_goalie_stats": _goalie_stats("away"),
            "home_goalie_stats": _goalie_stats("home"),
            "away_status": _status("away"),
            "home_status": _status("home"),
            "away_notes": game.get("awayNewsDetails") or game.get("awayNewsFantasyDetails"),
            "home_notes": game.get("homeNewsDetails") or game.get("homeNewsFantasyDetails"),
            "away_news_source": game.get("awayNewsSourceName"),
            "home_news_source": game.get("homeNewsSourceName"),
        }
        matchups.append(matchup)

    logger.info("Parsed %d matchups from Daily Faceoff (__NEXT_DATA__)", len(matchups))
    return matchups


def get_line_combinations(team_slug: str) -> dict:
    """Get line combinations for a team. NEVER cached.
    team_slug: lowercase with hyphens, e.g. 'colorado-avalanche'."""
    url = f"{BASE_URL}/teams/{team_slug}/line-combinations"
    soup = _fetch_soup(url)

    lines = {"forwards": [], "defense": [], "powerplay": [], "penalty_kill": []}

    tables = soup.find_all("table")
    current_section = "forwards"

    for table in tables:
        prev = table.find_previous(["h2", "h3", "h4"])
        if prev:
            header_text = prev.get_text(strip=True).lower()
            if "forward" in header_text or "lw" in header_text:
                current_section = "forwards"
            elif "defen" in header_text:
                current_section = "defense"
            elif "power" in header_text:
                current_section = "powerplay"
            elif "penalty" in header_text or "pk" in header_text:
                current_section = "penalty_kill"

        rows = table.find_all("tr")
        for row in rows:
            cells = row.find_all(["td", "th"])
            line_data = [c.get_text(strip=True) for c in cells if c.get_text(strip=True)]
            if line_data:
                lines[current_section].append(line_data)

    return lines


# Team slug mapping
TEAM_SLUGS = {
    "ANA": "anaheim-ducks", "ARI": "utah-hockey-club", "BOS": "boston-bruins",
    "BUF": "buffalo-sabres", "CGY": "calgary-flames", "CAR": "carolina-hurricanes",
    "CHI": "chicago-blackhawks", "COL": "colorado-avalanche", "CBJ": "columbus-blue-jackets",
    "DAL": "dallas-stars", "DET": "detroit-red-wings", "EDM": "edmonton-oilers",
    "FLA": "florida-panthers", "LAK": "los-angeles-kings", "MIN": "minnesota-wild",
    "MTL": "montreal-canadiens", "NSH": "nashville-predators", "NJD": "new-jersey-devils",
    "NYI": "new-york-islanders", "NYR": "new-york-rangers", "OTT": "ottawa-senators",
    "PHI": "philadelphia-flyers", "PIT": "pittsburgh-penguins", "SJS": "san-jose-sharks",
    "SEA": "seattle-kraken", "STL": "st-louis-blues", "TBL": "tampa-bay-lightning",
    "TOR": "toronto-maple-leafs", "VAN": "vancouver-canucks", "VGK": "vegas-golden-knights",
    "WPG": "winnipeg-jets", "WSH": "washington-capitals",
}
