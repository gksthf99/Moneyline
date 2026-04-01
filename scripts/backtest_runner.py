#!/usr/bin/env python3
"""
Backtest Runner v2 — full-season NBA + NHL prediction backtest.

Changes from v1:
  - NHL uses points percentage from official standings API (not W-L win rate)
  - Logit shrinkage applied to compress overconfident extremes
  - home_strength / away_strength columns track actual metric used

Usage:
    python scripts/backtest_runner.py
    python scripts/backtest_runner.py --start 2025-12-01
    python scripts/backtest_runner.py --start 2026-01-01 --end 2026-02-01
    python scripts/backtest_runner.py --dry-run
    python scripts/backtest_runner.py --resume
    python scripts/backtest_runner.py --debug
"""

import argparse
import csv
import json
import math
import os
import sys
import time
import logging
from datetime import date, datetime, timedelta
from pathlib import Path

import requests
import pandas as pd
from io import StringIO

# ---------------------------------------------------------------------------
# Project root setup
# ---------------------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

# Load .env
from dotenv import load_dotenv
load_dotenv(PROJECT_DIR / ".env")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger("backtest")


def setup_logging(debug: bool = False):
    level = logging.DEBUG if debug else logging.INFO
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-5s %(message)s", "%H:%M:%S"))
    logger.setLevel(level)
    logger.addHandler(handler)


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------

_last_request_time = 0.0
MIN_REQUEST_INTERVAL = 0.35


def rate_limited_get(url: str, max_retries: int = 3) -> requests.Response | None:
    """GET with rate limiting, retry on 429 and connection errors."""
    global _last_request_time
    for attempt in range(max_retries):
        elapsed = time.time() - _last_request_time
        if elapsed < MIN_REQUEST_INTERVAL:
            time.sleep(MIN_REQUEST_INTERVAL - elapsed)
        _last_request_time = time.time()

        try:
            resp = requests.get(url, timeout=15)
            if resp.status_code == 429:
                logger.warning("Rate limited (429), sleeping 10s...")
                time.sleep(10)
                continue
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return resp
        except requests.ConnectionError:
            wait = 2 ** attempt
            logger.warning("Connection error (attempt %d/%d), retrying in %ds...", attempt + 1, max_retries, wait)
            time.sleep(wait)
        except requests.HTTPError as e:
            logger.error("HTTP error: %s", e)
            return None
    logger.error("Failed after %d retries: %s", max_retries, url)
    return None


# ---------------------------------------------------------------------------
# Data fetchers
# ---------------------------------------------------------------------------

def fetch_nba_games(game_date: date) -> list[dict]:
    """Fetch NBA games for a date from ESPN scoreboard API."""
    url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={game_date.strftime('%Y%m%d')}"
    resp = rate_limited_get(url)
    if resp is None:
        return []

    games = []
    for event in resp.json().get("events", []):
        comp = event.get("competitions", [{}])[0]
        competitors = comp.get("competitors", [])
        if len(competitors) < 2:
            continue

        home = away = None
        for c in competitors:
            if c.get("homeAway") == "home":
                home = c
            else:
                away = c
        if not home or not away:
            continue

        is_final = comp.get("status", {}).get("type", {}).get("completed", False)

        home_score = None
        away_score = None
        if is_final:
            try:
                home_score = int(home.get("score", 0))
                away_score = int(away.get("score", 0))
            except (ValueError, TypeError):
                pass

        games.append({
            "home_abbr": home.get("team", {}).get("abbreviation", ""),
            "away_abbr": away.get("team", {}).get("abbreviation", ""),
            "home_score": home_score,
            "away_score": away_score,
            "is_final": is_final,
        })

    logger.debug("  NBA %s: %d games", game_date, len(games))
    return games


def fetch_nhl_games(game_date: date) -> list[dict]:
    """Fetch NHL games for a date from NHLe API."""
    url = f"https://api-web.nhle.com/v1/score/{game_date.isoformat()}"
    resp = rate_limited_get(url)
    if resp is None:
        return []

    games = []
    for game in resp.json().get("games", []):
        state = game.get("gameState", "")
        is_final = state in ("OFF", "FINAL") or state.startswith("OFF")

        home_score = None
        away_score = None
        if is_final:
            home_score = game.get("homeTeam", {}).get("score")
            away_score = game.get("awayTeam", {}).get("score")

        games.append({
            "home_abbr": game.get("homeTeam", {}).get("abbrev", ""),
            "away_abbr": game.get("awayTeam", {}).get("abbrev", ""),
            "home_score": home_score,
            "away_score": away_score,
            "is_final": is_final,
        })

    logger.debug("  NHL %s: %d games", game_date, len(games))
    return games


def fetch_nhl_standings(game_date: date) -> tuple[dict[str, float], dict[str, float]]:
    """Fetch NHL standings for a date.

    Returns (points_pct_dict, gf_ga_ratio_dict).
    Called ONCE per date. Returns empty dicts on failure.
    """
    url = f"https://api-web.nhle.com/v1/standings/{game_date.isoformat()}"
    resp = rate_limited_get(url)
    if resp is None:
        return {}, {}

    pts_result = {}
    gd_result = {}
    for team in resp.json().get("standings", []):
        abbrev_data = team.get("teamAbbrev", {})
        abbrev = abbrev_data.get("default", "") if isinstance(abbrev_data, dict) else str(abbrev_data)
        if not abbrev:
            continue
        wins = team.get("wins", 0)
        losses = team.get("losses", 0)
        ot_losses = team.get("otLosses", 0)
        pts_result[abbrev] = points_pct_smoothed(wins, losses, ot_losses)

        gf = team.get("goalsFor", 0)
        ga = team.get("goalsAgainst", 0)
        gd_result[abbrev] = gf / (gf + ga) if (gf + ga) > 0 else 0.5

    logger.debug("  NHL standings %s: %d teams", game_date, len(pts_result))
    return pts_result, gd_result


# ---------------------------------------------------------------------------
# Natural Stat Trick — xGF% (fetched once per backtest run)
# ---------------------------------------------------------------------------

# NST uses full team names; map to NHL API abbreviations
_NST_ABBREVS = {
    "Anaheim Ducks": "ANA", "Boston Bruins": "BOS", "Buffalo Sabres": "BUF",
    "Calgary Flames": "CGY", "Carolina Hurricanes": "CAR",
    "Chicago Blackhawks": "CHI", "Colorado Avalanche": "COL",
    "Columbus Blue Jackets": "CBJ", "Dallas Stars": "DAL",
    "Detroit Red Wings": "DET", "Edmonton Oilers": "EDM",
    "Florida Panthers": "FLA", "Los Angeles Kings": "LAK", "L.A. Kings": "LAK",
    "Minnesota Wild": "MIN", "Montréal Canadiens": "MTL",
    "Montreal Canadiens": "MTL", "Nashville Predators": "NSH",
    "New Jersey Devils": "NJD", "New York Islanders": "NYI",
    "New York Rangers": "NYR", "Ottawa Senators": "OTT",
    "Philadelphia Flyers": "PHI", "Pittsburgh Penguins": "PIT",
    "San Jose Sharks": "SJS", "Seattle Kraken": "SEA",
    "St Louis Blues": "STL", "St. Louis Blues": "STL",
    "Tampa Bay Lightning": "TBL", "Toronto Maple Leafs": "TOR",
    "Utah Hockey Club": "UTA", "Utah Mammoth": "UTA",
    "Vancouver Canucks": "VAN", "Vegas Golden Knights": "VGK",
    "Washington Capitals": "WSH", "Winnipeg Jets": "WPG",
}


def fetch_nst_xgf() -> dict[str, float]:
    """Fetch xGF% for all NHL teams from Natural Stat Trick (current season).

    Returns {team_abbrev: xgf_pct} where xgf_pct is 0.0-1.0.
    Called once at start. Uses season-wide data (mild look-ahead caveat).
    """
    url = "https://www.naturalstattrick.com/teamtable.php"
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
        "Accept": "text/html,application/xhtml+xml",
        "Accept-Language": "en-US,en;q=0.9",
    }

    try:
        resp = requests.get(url, headers=headers, timeout=20)
        resp.raise_for_status()
        tables = pd.read_html(StringIO(resp.text))
        if not tables:
            logger.warning("NST: no tables found")
            return {}
        df = tables[0]
    except Exception as e:
        logger.warning("NST fetch failed (continuing without xGF%%): %s", e)
        return {}

    result = {}
    for _, row in df.iterrows():
        team_name = str(row.get("Team", ""))
        xgf_pct = row.get("xGF%")
        if team_name and xgf_pct is not None:
            abbrev = _NST_ABBREVS.get(team_name, "")
            if abbrev:
                try:
                    result[abbrev] = float(xgf_pct) / 100.0
                except (ValueError, TypeError):
                    pass

    logger.info("NST xGF%% loaded: %d teams", len(result))
    return result


# ---------------------------------------------------------------------------
# H2H tracking (zero look-ahead — built incrementally during backtest)
# ---------------------------------------------------------------------------

def h2h_record_result(h2h: dict, team_a: str, team_b: str, winner: str):
    """Record a game result for H2H tracking."""
    key = frozenset([team_a, team_b])
    if key not in h2h:
        h2h[key] = {}
    h2h[key].setdefault(team_a, 0)
    h2h[key].setdefault(team_b, 0)
    if winner in h2h[key]:
        h2h[key][winner] += 1


def h2h_adjustment(h2h: dict, home_abbr: str, away_abbr: str) -> float:
    """Get H2H probability adjustment for home team.

    Matches production: ±3% cap, weighted by sample size (2g=50%, 4+=100%).
    Returns adjustment in probability space.
    """
    key = frozenset([home_abbr, away_abbr])
    if key not in h2h:
        return 0.0

    rec = h2h[key]
    home_wins = rec.get(home_abbr, 0)
    away_wins = rec.get(away_abbr, 0)
    total = home_wins + away_wins

    if total < 2:
        return 0.0

    home_h2h_pct = home_wins / total
    raw = home_h2h_pct - 0.5
    weight = min(1.0, total / 4)
    adj = max(-0.03, min(0.03, raw * weight))

    return adj if abs(adj) >= 0.005 else 0.0


# ---------------------------------------------------------------------------
# Probability model
# ---------------------------------------------------------------------------

def logit(p: float) -> float:
    p = max(0.001, min(0.999, p))
    return math.log(p / (1 - p))


def logistic(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def win_rate(wins: int, losses: int) -> float:
    """Win rate with Laplace smoothing (1W + 1L prior)."""
    return (wins + 1) / (wins + losses + 2)


def points_pct_smoothed(wins: int, losses: int, ot_losses: int) -> float:
    """NHL points percentage with Laplace smoothing.

    Points: W=2, OTL=1, L=0. Prior: 1W+1L equivalent (2 pts earned / 4 max).
    """
    games_played = wins + losses + ot_losses
    points_earned = (wins * 2) + (ot_losses * 1)
    return (points_earned + 2) / ((games_played * 2) + 4)


HOME_BONUS = {"nba": 0.363, "nhl": 0.200}

SHRINKAGE = {"nba": 0.70, "nhl": 1.00}

# NHL feature blending weights (match production baseline.py)
NHL_XG_WEIGHT = 0.30   # 70% points% + 30% xGF%
NHL_GD_WEIGHT = 0.10   # 90% blended + 10% GF/GA ratio

# Suppress home-ice bonus when away team's blended strength exceeds home by >5%
# (Lesson 8 fix — home favorites in 51-65% band losing to stronger away teams)
SUPPRESSION_THRESHOLD = 0.05


def layer1_prob(sport: str, home_strength: float, away_strength: float) -> float:
    """Bradley-Terry base probability with home advantage and logit shrinkage."""
    raw_lo = logit(home_strength) - logit(away_strength) + HOME_BONUS[sport]
    shrunk_lo = raw_lo * SHRINKAGE[sport]
    return logistic(shrunk_lo)


B2B_PENALTY = {"nba": 0.18, "nhl": 0.14}
REST_PER_DAY = {"nba": 0.06, "nhl": 0.04}
MAX_REST_DAYS = 3
SITUATIONAL_WEIGHT = {"nba": 0.25, "nhl": 1.00}


def layer2_adj(sport: str, home_b2b: bool, away_b2b: bool,
               home_rest: int | None, away_rest: int | None) -> float:
    """Situational adjustment in log-odds space."""
    adj = 0.0
    if home_b2b and not away_b2b:
        adj -= B2B_PENALTY[sport]
    elif away_b2b and not home_b2b:
        adj += B2B_PENALTY[sport]

    def extra(rest):
        return min(max((rest or 1) - 1, 0), MAX_REST_DAYS)

    adj += (extra(home_rest) - extra(away_rest)) * REST_PER_DAY[sport]
    return adj


def compute_probs(sport: str, home_strength: float, away_strength: float,
                  home_b2b: bool, away_b2b: bool,
                  home_rest: int | None, away_rest: int | None,
                  home_ice_suppressed: bool = False,
                  h2h_adj_prob: float = 0.0):
    """Compute base (shrunk), adjustment, and final probability.

    home_ice_suppressed: suppress NHL home bonus (Lesson 8).
    h2h_adj_prob: H2H adjustment in probability space (±0.03 max).
    """
    # Layer 1: shrunk base
    bonus = HOME_BONUS[sport]
    if sport == "nhl" and home_ice_suppressed:
        bonus = 0.0
    raw_lo = logit(home_strength) - logit(away_strength) + bonus
    shrunk_lo = raw_lo * SHRINKAGE[sport]
    base = logistic(shrunk_lo)

    # Layer 2: situational in log-odds after shrinkage
    adj = layer2_adj(sport, home_b2b, away_b2b, home_rest, away_rest)
    final = logistic(shrunk_lo + (adj * SITUATIONAL_WEIGHT[sport]))

    # Layer 2b: H2H in probability space (match production)
    final += h2h_adj_prob
    final = max(0.05, min(0.95, final))

    return base, adj, final


def assign_triage(final_prob: float) -> str:
    deviation = abs(final_prob - 0.5)
    if deviation > 0.20:
        return "skip"
    if deviation < 0.07:
        return "deep"
    return "standard"


def brier(predicted: float, outcome_bool: bool) -> float:
    return (predicted - int(outcome_bool)) ** 2


# ---------------------------------------------------------------------------
# Team state tracking (zero look-ahead)
# ---------------------------------------------------------------------------

class TeamState:
    __slots__ = ("wins", "losses", "last_game_date")

    def __init__(self):
        self.wins: int = 0
        self.losses: int = 0
        self.last_game_date: date | None = None

    def win_rate(self) -> float:
        return win_rate(self.wins, self.losses)

    def is_b2b(self, today: date) -> bool:
        if self.last_game_date is None:
            return False
        return self.last_game_date == today - timedelta(days=1)

    def rest_days(self, today: date) -> int | None:
        if self.last_game_date is None:
            return None
        return (today - self.last_game_date).days


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

CHECKPOINT_PATH = PROJECT_DIR / "backtest_results" / "checkpoint.json"


def load_checkpoint() -> set[str]:
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH) as f:
            data = json.load(f)
        return set(data.get("processed_dates", []))
    return set()


def save_checkpoint(processed: set[str]):
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump({"processed_dates": sorted(processed)}, f)


def delete_checkpoint():
    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()


# ---------------------------------------------------------------------------
# Supabase writer
# ---------------------------------------------------------------------------

def get_db_connection():
    import psycopg2
    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=int(os.getenv("DB_PORT", 5432)),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
    )


def flush_to_db(rows: list[dict], dry_run: bool):
    """Batch insert rows into backtest_results. Skips if dry_run."""
    if dry_run or not rows:
        return
    try:
        from psycopg2.extras import execute_values
        conn = get_db_connection()
        conn.autocommit = True
        cur = conn.cursor()

        cols = [
            "sport", "game_date", "home_team", "away_team",
            "home_wins", "home_losses", "away_wins", "away_losses",
            "home_win_rate", "away_win_rate",
            "home_b2b", "away_b2b", "home_rest_days", "away_rest_days",
            "base_prob_home", "situational_adj", "final_prob_home",
            "triage_tier",
            "home_score", "away_score", "actual_home_win", "brier_score",
            "home_strength", "away_strength",
        ]
        values = []
        for r in rows:
            values.append(tuple(r.get(c) for c in cols))

        sql = f"""
            INSERT INTO backtest_results ({', '.join(cols)})
            VALUES %s
            ON CONFLICT DO NOTHING
        """
        execute_values(cur, sql, values)
        cur.close()
        conn.close()
    except Exception as e:
        logger.error("DB write failed (continuing): %s", e)


# ---------------------------------------------------------------------------
# Main backtest loop
# ---------------------------------------------------------------------------

CSV_COLUMNS = [
    "sport", "game_date", "home_team", "away_team",
    "home_wins", "home_losses", "away_wins", "away_losses",
    "home_win_rate", "away_win_rate",
    "home_strength", "away_strength",
    "home_b2b", "away_b2b", "home_rest_days", "away_rest_days",
    "base_prob_home", "situational_adj", "final_prob_home",
    "shrinkage_factor",
    "triage_tier",
    "home_score", "away_score", "actual_home_win", "brier_score",
    # v3 columns: NHL feature stack
    "home_xgf_pct", "away_xgf_pct",
    "home_gf_ga_ratio", "away_gf_ga_ratio",
    "home_ice_suppressed", "h2h_adj",
]

SEASON_START = date(2025, 10, 1)


def run_backtest(start: date, end: date, dry_run: bool, resume: bool):
    """Run the full backtest loop."""
    out_dir = PROJECT_DIR / "backtest_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"nba_nhl_backtest_{ts}.csv"

    # Checkpoint
    processed = load_checkpoint() if resume else set()
    if resume and processed:
        logger.info("Resuming — %d dates already processed", len(processed))

    # Team state must be rebuilt from SEASON_START even when resuming
    nba_states: dict[str, TeamState] = {}
    nhl_states: dict[str, TeamState] = {}

    # NHL feature data (fetched once)
    nst_xgf = fetch_nst_xgf()

    # H2H tracking (built incrementally — zero look-ahead)
    nhl_h2h: dict = {}
    nba_h2h: dict = {}

    total_days = (end - SEASON_START).days + 1
    nba_count = 0
    nhl_count = 0
    nba_brier_sum = 0.0
    nhl_brier_sum = 0.0
    nba_graded = 0
    nhl_graded = 0
    nba_triage = {"skip": 0, "standard": 0, "deep": 0}
    nhl_triage = {"skip": 0, "standard": 0, "deep": 0}

    db_buffer: list[dict] = []

    with open(csv_path, "w", newline="") as csv_f:
        writer = csv.DictWriter(csv_f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        csv_f.flush()

        current = SEASON_START
        day_num = 0
        while current <= end:
            day_num += 1
            is_in_range = current >= start
            is_processed = current.isoformat() in processed

            # Fetch NHL standings ONCE per date (for points% + GF/GA lookup)
            nhl_standings_today: dict[str, float] = {}
            nhl_gd_today: dict[str, float] = {}
            nhl_standings_fetched = False

            for sport, fetch_fn, states in [
                ("nba", fetch_nba_games, nba_states),
                ("nhl", fetch_nhl_games, nhl_states),
            ]:
                games = fetch_fn(current)

                # Fetch NHL standings once when we first process NHL games
                if sport == "nhl" and games and not nhl_standings_fetched:
                    nhl_standings_today, nhl_gd_today = fetch_nhl_standings(current)
                    nhl_standings_fetched = True

                for game in games:
                    h_abbr = game["home_abbr"]
                    a_abbr = game["away_abbr"]
                    if not h_abbr or not a_abbr:
                        continue

                    # Ensure state exists
                    if h_abbr not in states:
                        states[h_abbr] = TeamState()
                    if a_abbr not in states:
                        states[a_abbr] = TeamState()

                    hs = states[h_abbr]
                    as_ = states[a_abbr]

                    # 1. Snapshot BEFORE this game
                    h_b2b = hs.is_b2b(current)
                    a_b2b = as_.is_b2b(current)
                    h_rest = hs.rest_days(current)
                    a_rest = as_.rest_days(current)
                    h_w, h_l = hs.wins, hs.losses
                    a_w, a_l = as_.wins, as_.losses

                    # Strength metric: NHL=blended (pts% + xGF% + GD), NBA=win_rate
                    h_xgf = None
                    a_xgf = None
                    h_gd = None
                    a_gd = None
                    suppressed = False
                    h2h_adj_val = 0.0

                    if sport == "nhl":
                        h_strength = nhl_standings_today.get(h_abbr, hs.win_rate())
                        a_strength = nhl_standings_today.get(a_abbr, as_.win_rate())

                        # Blend 70% points% + 30% xGF% (matches production baseline.py)
                        h_xgf = nst_xgf.get(h_abbr)
                        a_xgf = nst_xgf.get(a_abbr)
                        if h_xgf is not None and a_xgf is not None:
                            h_strength = (1 - NHL_XG_WEIGHT) * h_strength + NHL_XG_WEIGHT * h_xgf
                            a_strength = (1 - NHL_XG_WEIGHT) * a_strength + NHL_XG_WEIGHT * a_xgf

                        # Blend 90% + 10% GF/GA ratio (matches production baseline.py)
                        h_gd = nhl_gd_today.get(h_abbr)
                        a_gd = nhl_gd_today.get(a_abbr)
                        if h_gd is not None and a_gd is not None:
                            h_strength = (1 - NHL_GD_WEIGHT) * h_strength + NHL_GD_WEIGHT * h_gd
                            a_strength = (1 - NHL_GD_WEIGHT) * a_strength + NHL_GD_WEIGHT * a_gd

                        # Suppress home-ice bonus when away team clearly stronger
                        suppressed = round(a_strength - h_strength, 10) > SUPPRESSION_THRESHOLD

                        # H2H adjustment (zero look-ahead — uses only prior meetings)
                        h2h_adj_val = h2h_adjustment(nhl_h2h, h_abbr, a_abbr)
                    else:
                        h_strength = hs.win_rate()
                        a_strength = as_.win_rate()

                        # NBA H2H
                        h2h_adj_val = h2h_adjustment(nba_h2h, h_abbr, a_abbr)

                    h_wr = hs.win_rate()
                    a_wr = as_.win_rate()

                    # 2. Compute prediction
                    base, adj, final = compute_probs(
                        sport, h_strength, a_strength,
                        h_b2b, a_b2b, h_rest, a_rest,
                        home_ice_suppressed=suppressed,
                        h2h_adj_prob=h2h_adj_val,
                    )
                    tier = assign_triage(final)

                    # 3. Grade
                    home_won = None
                    brier_val = None
                    if game["is_final"] and game["home_score"] is not None and game["away_score"] is not None:
                        home_won = game["home_score"] > game["away_score"]
                        brier_val = brier(final, home_won)

                    # 4. Update state + H2H record
                    if game["is_final"] and game["home_score"] is not None:
                        winner = h_abbr if game["home_score"] > game["away_score"] else a_abbr
                        if game["home_score"] > game["away_score"]:
                            hs.wins += 1
                            as_.losses += 1
                        else:
                            hs.losses += 1
                            as_.wins += 1
                        # Record H2H (zero look-ahead — result recorded AFTER prediction)
                        sport_h2h = nhl_h2h if sport == "nhl" else nba_h2h
                        h2h_record_result(sport_h2h, h_abbr, a_abbr, winner)
                    hs.last_game_date = current
                    as_.last_game_date = current

                    # Only write CSV/DB if in range and not already processed
                    if not is_in_range or is_processed:
                        continue

                    row = {
                        "sport": sport,
                        "game_date": current.isoformat(),
                        "home_team": h_abbr,
                        "away_team": a_abbr,
                        "home_wins": h_w,
                        "home_losses": h_l,
                        "away_wins": a_w,
                        "away_losses": a_l,
                        "home_win_rate": round(h_wr, 4),
                        "away_win_rate": round(a_wr, 4),
                        "home_strength": round(h_strength, 4),
                        "away_strength": round(a_strength, 4),
                        "home_b2b": h_b2b,
                        "away_b2b": a_b2b,
                        "home_rest_days": h_rest,
                        "away_rest_days": a_rest,
                        "base_prob_home": round(base, 4),
                        "situational_adj": round(adj, 4),
                        "final_prob_home": round(final, 4),
                        "shrinkage_factor": SHRINKAGE[sport],
                        "triage_tier": tier,
                        "home_score": game["home_score"],
                        "away_score": game["away_score"],
                        "actual_home_win": home_won,
                        "brier_score": round(brier_val, 4) if brier_val is not None else None,
                        "home_xgf_pct": round(h_xgf, 4) if h_xgf is not None else None,
                        "away_xgf_pct": round(a_xgf, 4) if a_xgf is not None else None,
                        "home_gf_ga_ratio": round(h_gd, 4) if h_gd is not None else None,
                        "away_gf_ga_ratio": round(a_gd, 4) if a_gd is not None else None,
                        "home_ice_suppressed": suppressed,
                        "h2h_adj": round(h2h_adj_val, 4) if h2h_adj_val else None,
                    }

                    writer.writerow(row)
                    csv_f.flush()

                    if sport == "nba":
                        nba_count += 1
                        nba_triage[tier] += 1
                        if brier_val is not None:
                            nba_brier_sum += brier_val
                            nba_graded += 1
                    else:
                        nhl_count += 1
                        nhl_triage[tier] += 1
                        if brier_val is not None:
                            nhl_brier_sum += brier_val
                            nhl_graded += 1

                    db_buffer.append(row)
                    if len(db_buffer) >= 50:
                        flush_to_db(db_buffer, dry_run)
                        db_buffer.clear()

            # Checkpoint after each date
            if is_in_range and not is_processed:
                processed.add(current.isoformat())
                save_checkpoint(processed)

            # Progress log every 10 days
            if day_num % 10 == 0:
                logger.info(
                    "Progress: day %d/%d  (%s)  NBA games so far: %d  NHL games so far: %d",
                    day_num, total_days, current, nba_count, nhl_count,
                )

            current += timedelta(days=1)

        # Flush remaining
        if db_buffer:
            flush_to_db(db_buffer, dry_run)

    # Clean completion
    delete_checkpoint()

    # Summary
    nba_mean = nba_brier_sum / nba_graded if nba_graded else 0
    nhl_mean = nhl_brier_sum / nhl_graded if nhl_graded else 0

    print()
    print(f"NBA:  {nba_count} games  |  mean Brier: {nba_mean:.4f}  |  skip={nba_triage['skip']}  std={nba_triage['standard']}  deep={nba_triage['deep']}")
    print(f"NHL:  {nhl_count} games  |  mean Brier: {nhl_mean:.4f}  |  skip={nhl_triage['skip']}  std={nhl_triage['standard']}  deep={nhl_triage['deep']}")
    print(f"CSV: {csv_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="NBA/NHL full-season backtest runner v2")
    parser.add_argument("--start", type=str, default="2025-10-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None, help="End date (YYYY-MM-DD), default=yesterday")
    parser.add_argument("--dry-run", action="store_true", help="Skip Supabase writes")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--debug", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    setup_logging(args.debug)

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end) if args.end else date.today() - timedelta(days=1)

    logger.info("Backtest v2: %s → %s (%d days)%s",
                start, end, (end - start).days + 1,
                " [DRY RUN]" if args.dry_run else "")

    run_backtest(start, end, args.dry_run, args.resume)


if __name__ == "__main__":
    main()
