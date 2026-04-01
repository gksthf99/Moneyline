"""
NHL advanced stats from Natural Stat Trick (naturalstattrick.com).

Scraped from public HTML tables via requests + pandas. No API. No auth.
Used for: Corsi%, xG, high-danger chances.

Note: The default page serves data for the current season at 5v5.
Passing too many query params can cause empty table bodies.
Use minimal params for reliable scraping.
"""

import requests
import pandas as pd
from io import StringIO
from datetime import date

from src.data import cache

BASE_URL = "https://www.naturalstattrick.com"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "en-US,en;q=0.9",
}


def _current_season() -> str:
    """Return season string like '20252026'."""
    today = date.today()
    if today.month >= 10:
        return f"{today.year}{today.year + 1}"
    else:
        return f"{today.year - 1}{today.year}"


def get_team_stats(situation: str = "5v5") -> pd.DataFrame:
    """Get all team stats for current season (Corsi, Fenwick, xG, scoring chances).

    situation: 'all', '5v5', 'pp', 'pk', 'ev', etc.
    Returns DataFrame with 72 columns including: Team, GP, TOI, W, L,
    CF, CA, CF%, xGF, xGA, xGF%, HDCF, HDCA, HDCF%, etc.
    """
    season = _current_season()

    cached = cache.get("team_advanced", "nst_team", season, situation)
    if cached is not None:
        return cached

    sit_map = {"all": "all", "5v5": "5v5", "ev": "ev", "pp": "pp", "pk": "pk"}
    sit_param = sit_map.get(situation, "5v5")

    # Minimal params — default page serves current season
    params = {"sit": sit_param} if sit_param != "5v5" else {}

    resp = requests.get(f"{BASE_URL}/teamtable.php", params=params,
                        headers=HEADERS, timeout=20)
    resp.raise_for_status()

    tables = pd.read_html(StringIO(resp.text))
    if not tables:
        return pd.DataFrame()

    df = tables[0]
    # Drop the rank column if present
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    cache.put("team_advanced", df, "nst_team", season, situation)
    return df


def get_team_corsi(team: str, situation: str = "5v5") -> dict | None:
    """Get Corsi stats for a specific team.
    Returns dict with CF, CA, CF%, or None if team not found."""
    df = get_team_stats(situation)
    if df.empty:
        return None

    team_upper = team.upper()
    mask = df["Team"].astype(str).str.upper().str.contains(team_upper)
    if not mask.any():
        return None

    row = df[mask].iloc[0]
    result = {"team": row["Team"]}
    for stat in ["CF", "CA", "CF%", "FF", "FA", "FF%"]:
        if stat in df.columns:
            result[stat] = row[stat]

    return result


def get_team_xg(team: str, situation: str = "5v5") -> dict | None:
    """Get expected goals stats for a specific team.
    Returns dict with xGF, xGA, xGF%, or None if team not found."""
    df = get_team_stats(situation)
    if df.empty:
        return None

    team_upper = team.upper()
    mask = df["Team"].astype(str).str.upper().str.contains(team_upper)
    if not mask.any():
        return None

    row = df[mask].iloc[0]
    result = {"team": row["Team"]}
    for stat in ["xGF", "xGA", "xGF%", "GF", "GA", "GF%"]:
        if stat in df.columns:
            result[stat] = row[stat]

    return result


def get_team_high_danger(team: str, situation: str = "5v5") -> dict | None:
    """Get high-danger chances for a specific team.
    Returns dict with HDCF, HDCA, HDCF%, or None if team not found."""
    df = get_team_stats(situation)
    if df.empty:
        return None

    team_upper = team.upper()
    mask = df["Team"].astype(str).str.upper().str.contains(team_upper)
    if not mask.any():
        return None

    row = df[mask].iloc[0]
    result = {"team": row["Team"]}
    for stat in ["HDCF", "HDCA", "HDCF%", "SCF", "SCA", "SCF%"]:
        if stat in df.columns:
            result[stat] = row[stat]

    return result
