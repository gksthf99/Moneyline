"""
Research Agent — full implementation.

Deterministic pipeline per game:
  1. Collect data (schedule, standings, injuries, goalies)
  2. Build 3-layer probability decomposition (base + situational + information)
  3. Generate rule-based summary (no LLM)
  4. Post structured report to Discord thread
  5. Store to Supabase research table

Tom (OpenClaw agent) adds deeper qualitative reasoning via heartbeat.
"""

import json
import logging
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import requests

from src.config import SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY
from src.data.nba_espn import (
    get_scoreboard as nba_scoreboard, get_injuries as nba_injuries,
    get_team_leaders_from_scoreboard, get_team_recent_form, get_team_id_by_name,
)
from src.data.nhl_api import get_schedule as nhl_schedule, get_standings as nhl_standings
from src.data.nhl_dailyfaceoff import get_starting_goalies
from src.data.freshness import FreshnessChecker
from src.model.baseline import nba_base_probability, nhl_base_probability
from src.model.situational import SituationalFactors, nba_situational, nhl_situational
from src.model.information import (
    PlayerTier, InjuryStatus, calculate_player_impact, calculate_information_edge,
)
from src.model.decomposition import decompose_nba, decompose_nhl, ProbabilityDecomposition
from src.model.edge import full_edge_calculation
from src.data.polymarket import get_market_prices
from src.data.tennis_api import build_match_data as build_tennis_match_data, get_schedule as tennis_schedule
from src.repositories.games_repository import GamesRepository
from src.repositories.research_repository import ResearchRepository
from src.repositories.snapshot_repository import SnapshotRepository
from src.services.game_prediction import build_prediction_record, predict_game
from src.services.tennis_prediction import persist_tennis_prediction, predict_tennis
from src.time_utils import (
    EASTERN_TZ,
    CENTRAL_TZ,
    current_sports_date,
    format_game_time,
)

logger = logging.getLogger(__name__)
_games_repo = GamesRepository()
_research_repo = ResearchRepository()
_snapshot_repo = SnapshotRepository()


def _fetch_nhl_injuries() -> list[dict]:
    """Fetch NHL injuries from ESPN API (same structure as NBA)."""
    try:
        resp = requests.get(
            "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/injuries",
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        injuries = []
        for team_obj in data.get("injuries", []):
            team_name = team_obj.get("displayName", "")
            for item in team_obj.get("injuries", []):
                status = item.get("type", {}).get("description") or item.get("status", "")
                injuries.append({
                    "player": item.get("athlete", {}).get("displayName"),
                    "team": team_name,
                    "status": status,
                    "description": item.get("longComment") or item.get("shortComment"),
                })
        return injuries
    except Exception as e:
        logger.warning("Failed to fetch NHL injuries from ESPN: %s", e)
        return []


# ---------------------------------------------------------------------------
# Session context — lessons + prior grading
# ---------------------------------------------------------------------------

def load_session_context() -> dict:
    """Read lessons.md and prior Performance Agent grading from Supabase."""
    context = {"lessons": "", "prior_grading": []}

    # Read lessons.md
    lessons_path = Path(__file__).resolve().parent.parent.parent / "tasks" / "lessons.md"
    if lessons_path.exists():
        context["lessons"] = lessons_path.read_text()

    # Query most recent calibration entries
    if SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY:
        try:
            resp = requests.get(
                f"{SUPABASE_URL}/rest/v1/calibration",
                headers={
                    "apikey": SUPABASE_SERVICE_ROLE_KEY,
                    "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
                },
                params={
                    "select": "date,brier_score_rolling,attribution_breakdown",
                    "order": "date.desc",
                    "limit": "5",
                },
                timeout=10,
            )
            if resp.status_code == 200:
                context["prior_grading"] = resp.json()
        except Exception as e:
            logger.warning("Failed to load prior grading: %s", e)

    return context


# ---------------------------------------------------------------------------
# Slate loading
# ---------------------------------------------------------------------------

def load_todays_slate() -> list[dict]:
    """Load today's games from Supabase games table.

    Uses a 6 AM Eastern sports-day boundary so late-night games stay on
    the intended slate across DST changes.
    """
    return _games_repo.list_slate()


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def _parse_record(summary: str) -> tuple[int, int]:
    """Parse 'W-L' record string into (wins, losses)."""
    try:
        parts = summary.split("-")
        return int(parts[0]), int(parts[1])
    except (ValueError, IndexError):
        return 0, 0


def _nhl_str(val) -> str:
    """Extract string from NHL API field (may be str or {'default': str})."""
    if isinstance(val, dict):
        return val.get("default", "")
    return val or ""


def collect_nba_data(home_team: str, away_team: str) -> dict:
    """Gather all NBA data for a game: records, splits, leaders, form, injuries."""
    data = {
        "sport": "NBA",
        "home_team": home_team,
        "away_team": away_team,
        "home_win_pct": 0.5,
        "away_win_pct": 0.5,
        "home_record": "",
        "away_record": "",
        "home_home_record": "",
        "home_away_record": "",
        "away_home_record": "",
        "away_away_record": "",
        "home_leaders": {},
        "away_leaders": {},
        "home_form": {},
        "away_form": {},
        "injuries": [],
        "freshness": {},
    }

    checker = FreshnessChecker()

    # Records + leaders + home/away splits — single scoreboard call
    try:
        raw = requests.get(
            "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard",
            params={"dates": date.today().strftime("%Y%m%d")},
            timeout=15,
        ).json()

        for event in raw.get("events", []):
            comp = event.get("competitions", [{}])[0]
            teams_in_game = {}
            for t in comp.get("competitors", []):
                name = t.get("team", {}).get("displayName", "")
                teams_in_game[name] = t

            if home_team in teams_in_game and away_team in teams_in_game:
                for team_name, prefix in [(home_team, "home"), (away_team, "away")]:
                    t = teams_in_game[team_name]
                    records = {}
                    for rec in t.get("records", []):
                        records[rec.get("type", "")] = rec.get("summary", "")

                    # Overall record + win%
                    total_rec = records.get("total", "")
                    if total_rec:
                        w, l = _parse_record(total_rec)
                        gp = w + l
                        data[f"{prefix}_win_pct"] = w / gp if gp > 0 else 0.5
                        data[f"{prefix}_record"] = total_rec

                    # Home/away splits — extracted from same data
                    data[f"{prefix}_home_record"] = records.get("home", "")
                    data[f"{prefix}_away_record"] = records.get("road", "")

                    # Leaders
                    leaders = {}
                    for leader in t.get("leaders", []):
                        key = leader.get("name", "").lower()
                        for a in leader.get("leaders", [])[:1]:
                            athlete = a.get("athlete", {})
                            leaders[key] = {
                                "name": athlete.get("displayName", "?"),
                                "value": a.get("displayValue", "?"),
                            }
                    data[f"{prefix}_leaders"] = leaders

                break

        checker.record("espn", "team_record", home_team, data["home_record"])
        checker.record("espn", "team_record", away_team, data["away_record"])
    except Exception as e:
        logger.warning("Failed to get NBA scoreboard data: %s", e)

    # Recent form (L10 + streak)
    for team_name, prefix in [(home_team, "home"), (away_team, "away")]:
        try:
            team_id = get_team_id_by_name(team_name)
            if team_id:
                data[f"{prefix}_form"] = get_team_recent_form(team_id)
        except Exception as e:
            logger.warning("Failed to get form for %s: %s", team_name, e)

    # Injuries
    try:
        injuries = nba_injuries()
        game_injuries = [
            inj for inj in injuries
            if inj.get("team") in (home_team, away_team)
        ]
        data["injuries"] = game_injuries

        for inj in game_injuries:
            checker.record("espn", "injury", inj.get("player", ""), inj.get("status", "unknown"))
    except Exception as e:
        logger.warning("Failed to get NBA injuries: %s", e)

    # NBA.com advanced ratings (ORtg, DRtg, NetRtg per 100 possessions)
    try:
        from src.data.nba_stats import get_team_ratings, get_team_ratings_recent, get_team_ratings_clutch
        season_ratings = get_team_ratings()
        l10_ratings = get_team_ratings_recent(10)
        clutch_ratings = get_team_ratings_clutch()

        if season_ratings:
            data["home_net_rtg"] = {
                "season": season_ratings.get(home_team, {}),
                "l10": l10_ratings.get(home_team),
                "clutch": clutch_ratings.get(home_team),
            }
            data["away_net_rtg"] = {
                "season": season_ratings.get(away_team, {}),
                "l10": l10_ratings.get(away_team),
                "clutch": clutch_ratings.get(away_team),
            }
    except Exception as e:
        logger.warning("Failed to get NBA.com ratings: %s", e)

    data["freshness"] = checker.get_timestamps()
    data["disagreements"] = [d.description for d in checker.check_all()]

    return data


def collect_nhl_data(home_team: str, away_team: str) -> dict:
    """Gather all NHL data: records, splits, L10, streak, goalies, advanced stats."""
    data = {
        "sport": "NHL",
        "home_team": home_team,
        "away_team": away_team,
        "home_gf": 0, "home_ga": 0, "home_gp": 0,
        "away_gf": 0, "away_ga": 0, "away_gp": 0,
        "home_record": "",
        "away_record": "",
        "home_home_record": "",
        "home_away_record": "",
        "away_home_record": "",
        "away_away_record": "",
        "home_l10": "",
        "away_l10": "",
        "home_streak": "",
        "away_streak": "",
        "home_goalie": None,
        "away_goalie": None,
        "home_advanced": {},
        "away_advanced": {},
        "injuries": [],
        "freshness": {},
    }

    checker = FreshnessChecker()

    def _match_team(standings_name: str, target: str) -> bool:
        if standings_name == target:
            return True
        if target.endswith(standings_name.split()[-1] if standings_name else ""):
            return True
        return False

    def _extract_team_data(team: dict, prefix: str):
        data[f"{prefix}_gf"] = team.get("goalFor", 0)
        data[f"{prefix}_ga"] = team.get("goalAgainst", 0)
        data[f"{prefix}_gp"] = team.get("gamesPlayed", 0)
        w, l, otl = team.get("wins", 0), team.get("losses", 0), team.get("otLosses", 0)
        data[f"{prefix}_record"] = f"{w}-{l}-{otl}"
        # Home/away splits
        hw, hl, hotl = team.get("homeWins", 0), team.get("homeLosses", 0), team.get("homeOtLosses", 0)
        rw, rl, rotl = team.get("roadWins", 0), team.get("roadLosses", 0), team.get("roadOtLosses", 0)
        data[f"{prefix}_home_record"] = f"{hw}-{hl}-{hotl}"
        data[f"{prefix}_away_record"] = f"{rw}-{rl}-{rotl}"
        # L10 + streak
        l10w, l10l, l10otl = team.get("l10Wins", 0), team.get("l10Losses", 0), team.get("l10OtLosses", 0)
        data[f"{prefix}_l10"] = f"{l10w}-{l10l}-{l10otl}"
        streak_code = team.get("streakCode", "")
        streak_count = team.get("streakCount", 0)
        data[f"{prefix}_streak"] = f"{streak_code}{streak_count}" if streak_code else ""

    # Standings — has everything: records, splits, L10, streak, goals
    try:
        standings_data = nhl_standings()
        for team in standings_data.get("standings", []):
            team_name = _nhl_str(team.get("teamName"))
            if _match_team(team_name, home_team):
                _extract_team_data(team, "home")
                checker.record("nhl_api", "record", home_team, data["home_record"])
            elif _match_team(team_name, away_team):
                _extract_team_data(team, "away")
                checker.record("nhl_api", "record", away_team, data["away_record"])
    except Exception as e:
        logger.warning("Failed to get NHL standings: %s", e)

    # Advanced stats from Natural Stat Trick
    try:
        from src.data.nhl_naturalstattrick import get_team_corsi, get_team_xg
        for team_name, prefix in [(home_team, "home"), (away_team, "away")]:
            # Use last word of team name for NST matching
            search = team_name.split()[-1] if team_name else ""
            corsi = get_team_corsi(search)
            xg = get_team_xg(search)
            adv = {}
            if corsi:
                adv["CF%"] = corsi.get("CF%", "")
            if xg:
                adv["xGF%"] = xg.get("xGF%", "")
                adv["GF"] = xg.get("GF", "")
                adv["GA"] = xg.get("GA", "")
            data[f"{prefix}_advanced"] = adv
    except Exception as e:
        logger.warning("Failed to get NST advanced stats: %s", e)

    # Goalie starts
    try:
        matchups = get_starting_goalies(current_sports_date())
        for m in matchups:
            if (home_team in m.get("home_team", "") or
                    m.get("home_team", "") in home_team):
                data["home_goalie"] = m.get("home_goalie")
                data["away_goalie"] = m.get("away_goalie")
                break
    except Exception as e:
        logger.warning("Failed to get goalie starts: %s", e)

    # Injuries (via ESPN NHL)
    try:
        injuries = _fetch_nhl_injuries()
        game_injuries = [
            inj for inj in injuries
            if inj.get("team") in (home_team, away_team)
        ]
        data["injuries"] = game_injuries
        for inj in game_injuries:
            checker.record("espn", "injury", inj.get("player", ""), inj.get("status", "unknown"))
    except Exception as e:
        logger.warning("Failed to get NHL injuries: %s", e)

    data["freshness"] = checker.get_timestamps()
    data["disagreements"] = [d.description for d in checker.check_all()]

    return data


# ---------------------------------------------------------------------------
# Situational factors from morning slate
# ---------------------------------------------------------------------------

def _get_situational_from_slate(game: dict) -> SituationalFactors:
    """Build SituationalFactors from the Supabase game record + morning slate data."""
    # Query morning slate data if available, otherwise use defaults
    # For now, pull B2B info by checking yesterday's schedule
    home = game.get("home_team", "")
    away = game.get("away_team", "")
    sport = game.get("sport", "").upper()

    # Check yesterday's games for B2B
    from src.scripts.morning_slate import (
        _get_nba_yesterday_teams, _get_nhl_yesterday_teams, _compute_travel_flag,
    )

    if sport == "NBA":
        yesterday_teams = _get_nba_yesterday_teams()
    else:
        yesterday_teams = _get_nhl_yesterday_teams()

    home_b2b = home in yesterday_teams
    away_b2b = away in yesterday_teams

    _, travel_zones = _compute_travel_flag(home, away, sport)

    return SituationalFactors(
        home_days_rest=1 if home_b2b else 2,
        away_days_rest=1 if away_b2b else 2,
        home_is_b2b=home_b2b,
        away_is_b2b=away_b2b,
        home_travel_zones=0,  # Home team doesn't travel
        away_travel_zones=travel_zones,
    )


# ---------------------------------------------------------------------------
# Injury → PlayerImpact mapping
# ---------------------------------------------------------------------------

# Status string → InjuryStatus mapping
_STATUS_MAP = {
    "out": InjuryStatus.OUT,
    "doubtful": InjuryStatus.DOUBTFUL,
    "questionable": InjuryStatus.QUESTIONABLE,
    "probable": InjuryStatus.PROBABLE,
    "active": InjuryStatus.ACTIVE,
    "day-to-day": InjuryStatus.QUESTIONABLE,
    "expected to play": InjuryStatus.PROBABLE,
    "injured reserve": InjuryStatus.OUT,
    "suspension": InjuryStatus.OUT,
    "out for season": InjuryStatus.OUT,
    "10-day dl": InjuryStatus.OUT,
    "15-day dl": InjuryStatus.OUT,
    "60-day dl": InjuryStatus.OUT,
}


def _tier_from_ppg(ppg: float) -> PlayerTier:
    """Assign player tier based on PPG. Used for auto-tiering injuries."""
    if ppg >= 20:
        return PlayerTier.TIER_1   # Star: >20 PPG → -6%
    elif ppg >= 15:
        return PlayerTier.TIER_2   # Starter: 15-20 PPG → -3.5%
    elif ppg >= 10:
        return PlayerTier.TIER_3   # Rotation: 10-15 PPG → -2%
    else:
        return PlayerTier.TIER_4   # Role player: <10 PPG → -0.75%


# Updated tier impacts to match the spec
_TIER_IMPACT_OVERRIDE = {
    PlayerTier.TIER_1: 0.06,   # -6%
    PlayerTier.TIER_2: 0.035,  # -3.5%
    PlayerTier.TIER_3: 0.02,   # -2%
    PlayerTier.TIER_4: 0.0075, # -0.75%
}


def _map_injuries_to_impacts(injuries: list[dict], home_team: str,
                               leaders: dict | None = None,
                               measured_impact: dict | None = None) -> list:
    """Convert raw injury dicts to PlayerImpact objects.

    Uses measured on/off court impact when available (R2 upgrade).
    Falls back to PPG-based tier system when measured data is missing.

    Args:
        injuries: Raw injury dicts from ESPN
        home_team: Home team display name
        leaders: {"player_name": ppg_float, ...} for tier fallback
        measured_impact: {"player_name": impact_float, ...} from nba_player_stats
    """
    impacts = []
    ppg_map = leaders or {}
    impact_map = measured_impact or {}

    for inj in injuries:
        player = inj.get("player", "")
        team = inj.get("team", "")
        status_str = (inj.get("status") or "").lower().strip()

        status = _STATUS_MAP.get(status_str, InjuryStatus.QUESTIONABLE)

        # Skip active/probable players
        if status in (InjuryStatus.ACTIVE, InjuryStatus.PROBABLE):
            continue

        side = "home" if team == home_team else "away"

        # R2: Use measured impact if available
        measured = impact_map.get(player)
        if measured is not None:
            # measured is the absolute impact (positive = player helps team)
            # Negative measured impact means team is better without them — cap at 0
            abs_impact = max(measured, 0.0)

            # Assign tier label based on measured impact for display purposes
            if abs_impact >= 0.06:
                tier = PlayerTier.TIER_1
            elif abs_impact >= 0.035:
                tier = PlayerTier.TIER_2
            elif abs_impact >= 0.015:
                tier = PlayerTier.TIER_3
            else:
                tier = PlayerTier.TIER_4

            impact = calculate_player_impact(
                player_name=player,
                team=side,
                tier=tier,
                status=status,
                impact_estimate=abs_impact,
            )
        else:
            # Fallback: PPG-based tier system
            ppg = ppg_map.get(player, 0)
            tier = _tier_from_ppg(ppg) if ppg > 0 else PlayerTier.TIER_3

            impact = calculate_player_impact(
                player_name=player,
                team=side,
                tier=tier,
                status=status,
                impact_estimate=_TIER_IMPACT_OVERRIDE.get(tier),
            )

        impacts.append(impact)

    return impacts


# ---------------------------------------------------------------------------
# Decomposition
# ---------------------------------------------------------------------------

def _parse_split_pct(record_str: str) -> float | None:
    """Parse a 'W-L' or 'W-L-OTL' record into win%. Returns None if unparseable."""
    if not record_str:
        return None
    parts = record_str.split("-")
    try:
        w = int(parts[0])
        l = int(parts[1])
        gp = w + l + (int(parts[2]) if len(parts) > 2 else 0)
        return w / gp if gp > 0 else None
    except (ValueError, IndexError):
        return None


def _parse_l10_pct(record_str: str) -> float | None:
    """Parse L10 record (e.g. '7-3' or '6-4-0') into win%."""
    return _parse_split_pct(record_str)


def _build_ppg_map(game_data: dict) -> dict:
    """Build player→PPG map from leader data for injury tiering."""
    ppg_map = {}
    for prefix in ("home", "away"):
        leaders = game_data.get(f"{prefix}_leaders", {})
        # The rating leader usually has PPG in the value string
        rating = leaders.get("rating", {})
        if rating.get("name") and rating.get("value"):
            # Parse "25.8 PPG, 5.5 RPG, ..." or just "25.8"
            val_str = str(rating["value"]).split(" ")[0].replace(",", "")
            try:
                ppg_map[rating["name"]] = float(val_str)
            except ValueError:
                pass
        # Also grab the points leader
        pts = leaders.get("pointspergame", leaders.get("points", {}))
        if pts.get("name") and pts.get("value"):
            try:
                ppg_map[pts["name"]] = float(str(pts["value"]).split(" ")[0])
            except ValueError:
                pass
    return ppg_map


# NHL top scorers cache — fetched once per session from NHL API
_nhl_ppg_cache: dict | None = None


def _build_nhl_ppg_map() -> dict:
    """Fetch NHL scoring leaders and estimate PPG for tier assignment.

    Uses points/82 as a rough PPG proxy since NHL API leaders endpoint
    doesn't include GP alongside points. Top 50 scorers covers all
    players likely to appear on injury reports that matter.
    """
    global _nhl_ppg_cache
    if _nhl_ppg_cache is not None:
        return _nhl_ppg_cache

    ppg_map = {}
    try:
        resp = requests.get(
            "https://api-web.nhle.com/v1/skater-stats-leaders/current?categories=points&limit=100",
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        for player in data.get("points", []):
            first = player.get("firstName", {}).get("default", "")
            last = player.get("lastName", {}).get("default", "")
            name = f"{first} {last}".strip()
            pts = player.get("value", 0)
            # Estimate PPG: use ~72 GP average for a full-season player
            # This is rough but sufficient for tiering (T1 >20 PPG ≈ >90 pts)
            # NHL "PPG" = points/game, roughly pts/72
            ppg_estimate = pts / 72 if pts > 0 else 0
            # Convert to NBA-equivalent PPG scale for tiering:
            # NHL T1 (>1.2 PPG ≈ >90pts): map to >20 NBA PPG
            # NHL T2 (>0.8 PPG ≈ >60pts): map to 15-20
            # NHL T3 (>0.5 PPG ≈ >36pts): map to 10-15
            # Scale: NHL PPG × 16.67 ≈ NBA PPG equivalent
            nba_equiv = ppg_estimate * 16.67
            ppg_map[name] = nba_equiv
        logger.info("Built NHL PPG map: %d players", len(ppg_map))
    except Exception as e:
        logger.warning("Failed to build NHL PPG map: %s", e)

    _nhl_ppg_cache = ppg_map
    return ppg_map


def build_decomposition(game: dict, game_data: dict) -> ProbabilityDecomposition:
    """Run the 3-layer probability decomposition for a game.

    Layer 1: Elo from home/away splits (not overall win%)
    Layer 2: B2B + rest + travel + recent form (60% L10 + 40% season)
    Layer 3: Injuries auto-tiered by PPG
    """
    sport = game.get("sport", "").upper()
    home = game.get("home_team", "")
    away = game.get("away_team", "")

    # Build situational factors with form data
    sit_factors = _get_situational_from_slate(game)
    game_data["_situational_inputs"] = {
        "home_days_rest": sit_factors.home_days_rest,
        "away_days_rest": sit_factors.away_days_rest,
        "home_is_b2b": sit_factors.home_is_b2b,
        "away_is_b2b": sit_factors.away_is_b2b,
        "home_travel_zones": sit_factors.home_travel_zones,
        "away_travel_zones": sit_factors.away_travel_zones,
    }

    # Inject form data into situational factors
    if sport == "NBA":
        home_form = game_data.get("home_form", {})
        away_form = game_data.get("away_form", {})
        sit_factors.home_l10_pct = _parse_l10_pct(home_form.get("record"))
        sit_factors.away_l10_pct = _parse_l10_pct(away_form.get("record"))
        sit_factors.home_season_pct = game_data.get("home_win_pct")
        sit_factors.away_season_pct = game_data.get("away_win_pct")
        # H2H
        try:
            from src.data.h2h import get_nba_h2h
            h2h = get_nba_h2h(home, away)
            if h2h["total_games"] >= 2:
                sit_factors.home_h2h_pct = h2h["home_h2h_pct"]
                sit_factors.h2h_games = h2h["total_games"]
                game_data["h2h"] = h2h
        except Exception as e:
            logger.warning("H2H fetch failed for %s vs %s: %s", home, away, e)
    else:
        sit_factors.home_l10_pct = _parse_l10_pct(game_data.get("home_l10"))
        sit_factors.away_l10_pct = _parse_l10_pct(game_data.get("away_l10"))
        home_gp = game_data.get("home_gp", 1)
        away_gp = game_data.get("away_gp", 1)
        home_w = int(game_data.get("home_record", "0-0-0").split("-")[0]) if game_data.get("home_record") else 0
        away_w = int(game_data.get("away_record", "0-0-0").split("-")[0]) if game_data.get("away_record") else 0
        sit_factors.home_season_pct = home_w / home_gp if home_gp else None
        sit_factors.away_season_pct = away_w / away_gp if away_gp else None
        # H2H
        try:
            from src.data.h2h import get_nhl_h2h, _nhl_abbrev
            home_abbrev = _nhl_abbrev(home)
            away_abbrev = _nhl_abbrev(away)
            h2h = get_nhl_h2h(home, away, home_abbrev, away_abbrev)
            if h2h["total_games"] >= 2:
                sit_factors.home_h2h_pct = h2h["home_h2h_pct"]
                sit_factors.h2h_games = h2h["total_games"]
                game_data["h2h"] = h2h
        except Exception as e:
            logger.warning("H2H fetch failed for %s vs %s: %s", home, away, e)
            home_abbrev = ""
            away_abbrev = ""

        # Goalie adjustment (NHL only)
        try:
            from src.data.nhl_goalies import goalie_adjustment
            from src.data.h2h import _nhl_abbrev as _abbrev
            h_abbr = home_abbrev if home_abbrev else _abbrev(home)
            a_abbr = away_abbrev if away_abbrev else _abbrev(away)
            home_goalie_name = game_data.get("home_goalie")
            away_goalie_name = game_data.get("away_goalie")
            g_adj, g_desc = goalie_adjustment(home_goalie_name, away_goalie_name, h_abbr, a_abbr)
            if g_adj != 0:
                sit_factors.goalie_adj_value = g_adj
                sit_factors.goalie_adj_desc = g_desc
                game_data["goalie_adjustment"] = {"value": g_adj, "desc": g_desc}
        except Exception as e:
            logger.warning("Goalie adjustment failed for %s vs %s: %s", home, away, e)

    # Build PPG map for injury tiering (fallback) + measured impact map (R2)
    measured_impact = None
    if sport == "NHL":
        ppg_map = _build_nhl_ppg_map()
    else:
        ppg_map = _build_ppg_map(game_data)
        # R2: Fetch measured on/off impact for NBA players
        try:
            from src.data.nba_player_stats import build_player_impact_map
            measured_impact = build_player_impact_map()
        except Exception as e:
            logger.warning("Failed to load measured player impact: %s", e)

    injuries = game_data.get("injuries", [])
    player_impacts = _map_injuries_to_impacts(injuries, home, leaders=ppg_map, measured_impact=measured_impact)

    # Pull market prices from Polymarket
    market = get_market_prices(home, away, sport)
    ask_price = None
    bid_price = None
    away_ask_price = None
    away_bid_price = None
    if market:
        # For edge calc: ask = what you pay to buy home team token
        # CLOB ask is the real price; fall back to Gamma price
        ask_price = market.get("home_ask") or market.get("home_price")
        bid_price = market.get("home_bid") or market.get("home_price")
        # Away side market data (for two-sided edge calculation)
        away_ask_price = market.get("away_ask") or market.get("away_price")
        away_bid_price = market.get("away_bid") or market.get("away_price")
        game_data["market"] = market

    # Rolling CLV health as a persisted feature for future calibration.
    try:
        from src.model.clv import check_clv_health
        clv = check_clv_health()
        if clv.get("sample_size", 0) >= 5:
            game_data["_clv_residual"] = clv.get("rolling_7d_clv_bps", 0) / 10000
    except Exception:
        pass

    # Hours to game (for threshold selection)
    hours_to_game = 6.0
    game_time = game.get("game_time", "")
    if game_time:
        try:
            gt = datetime.fromisoformat(game_time.replace("Z", "+00:00"))
            hours_to_game = max(0, (gt - datetime.now(timezone.utc)).total_seconds() / 3600)
        except (ValueError, TypeError):
            pass

    # Parse home/away split win%
    home_home_pct = _parse_split_pct(game_data.get("home_home_record"))
    away_away_pct = _parse_split_pct(game_data.get("away_away_record"))

    if sport == "NBA":
        # When net rating data available, disable L2 form adjustment
        # to avoid double-counting (form is already in the 0.4 recent weight)
        home_net_rtg = game_data.get("home_net_rtg")
        away_net_rtg = game_data.get("away_net_rtg")
        if home_net_rtg and home_net_rtg.get("season"):
            sit_factors.home_l10_pct = None
            sit_factors.away_l10_pct = None

        return decompose_nba(
            home_team=home,
            away_team=away,
            home_win_pct=game_data.get("home_win_pct", 0.5),
            away_win_pct=game_data.get("away_win_pct", 0.5),
            home_home_pct=home_home_pct,
            away_away_pct=away_away_pct,
            situational=sit_factors,
            player_impacts=player_impacts,
            ask_price=ask_price,
            bid_price=bid_price,
            hours_to_game=hours_to_game,
            home_net_rtg=home_net_rtg,
            away_net_rtg=away_net_rtg,
            away_ask_price=away_ask_price,
            away_bid_price=away_bid_price,
        )
    else:
        # Parse NHL win% from records
        def _nhl_win_pct(record_str: str) -> float:
            """Parse 'W-L-OTL' → win%."""
            if not record_str:
                return 0.5
            parts = record_str.split("-")
            if len(parts) < 2:
                return 0.5
            try:
                w = int(parts[0])
                total = sum(int(p) for p in parts)
                return w / total if total > 0 else 0.5
            except (ValueError, ZeroDivisionError):
                return 0.5

        home_win_pct = _nhl_win_pct(game_data.get("home_record", ""))
        away_win_pct = _nhl_win_pct(game_data.get("away_record", ""))
        home_home_pct = _nhl_win_pct(game_data.get("home_home_record", "")) if game_data.get("home_home_record") else None
        away_away_pct = _nhl_win_pct(game_data.get("away_away_record", "")) if game_data.get("away_away_record") else None

        # Extract xGF% from advanced stats (from Natural Stat Trick)
        home_xgf = game_data.get("home_advanced", {}).get("xGF%")
        away_xgf = game_data.get("away_advanced", {}).get("xGF%")
        home_xgf_pct = float(home_xgf) / 100.0 if home_xgf else None
        away_xgf_pct = float(away_xgf) / 100.0 if away_xgf else None

        # Goal differential ratio: GF / (GF + GA)
        home_gf = game_data.get("home_gf", 0)
        home_ga = game_data.get("home_ga", 0)
        away_gf = game_data.get("away_gf", 0)
        away_ga = game_data.get("away_ga", 0)
        home_gf_ga = home_gf / (home_gf + home_ga) if (home_gf + home_ga) > 0 else None
        away_gf_ga = away_gf / (away_gf + away_ga) if (away_gf + away_ga) > 0 else None

        return decompose_nhl(
            home_team=home,
            away_team=away,
            home_win_pct=home_win_pct,
            away_win_pct=away_win_pct,
            home_home_pct=home_home_pct,
            away_away_pct=away_away_pct,
            home_xgf_pct=home_xgf_pct,
            away_xgf_pct=away_xgf_pct,
            home_gf_ga_ratio=home_gf_ga,
            away_gf_ga_ratio=away_gf_ga,
            situational=sit_factors,
            player_impacts=player_impacts,
            ask_price=ask_price,
            bid_price=bid_price,
            hours_to_game=hours_to_game,
            away_ask_price=away_ask_price,
            away_bid_price=away_bid_price,
        )


# ---------------------------------------------------------------------------
# Deterministic summary — no LLM, rule-based from data
# Tom (OpenClaw agent) adds deeper reasoning later via heartbeat
# ---------------------------------------------------------------------------

def build_summary(
    decomposition: ProbabilityDecomposition,
    game_data: dict,
) -> str:
    """Generate a rule-based research summary from collected data.

    Zero LLM cost. Includes: records, splits, form, leaders, injuries, layers, market, edge.
    """
    sport = game_data.get("sport", "NBA")
    home = decomposition.home_team
    away = decomposition.away_team
    home_prob = decomposition.final_probability
    away_prob = 1 - home_prob
    sit = decomposition.situational_adjustment
    info = decomposition.information_edge

    # Determine pick: use bet_side when two-sided edge is active
    if decomposition.bet_side == "away":
        pick, pick_prob = away, away_prob
    elif decomposition.bet_side == "home":
        pick, pick_prob = home, home_prob
    elif home_prob >= 0.5:
        pick, pick_prob = home, home_prob
    else:
        pick, pick_prob = away, away_prob

    # Derive recommendation from edge data
    # When two-sided edge is active, bet_side is the authority — it enforces
    # both edge threshold AND model conviction floor (40%).
    has_any_edge = decomposition.edge is not None or decomposition.away_edge is not None
    if decomposition.bet_side is not None:
        rec = "BET"
        rec_label = f"BET — {decomposition.bet_side} side passes threshold"
    elif has_any_edge:
        active_edge = decomposition.edge or decomposition.away_edge
        rec = "PASS"
        rec_label = f"PASS — below threshold or model conviction"
    else:
        rec = "MONITOR"
        rec_label = "MONITOR — no market data"

    lines = [
        f"**Recommendation: {rec_label}**",
        f"**Predicted winner: {pick} ({pick_prob:.0%})**",
    ]

    # --- Records & Splits ---
    lines.append("")
    lines.append("**Records**")
    home_rec = game_data.get("home_record", "?")
    away_rec = game_data.get("away_record", "?")
    home_home = game_data.get("home_home_record", "")
    home_away = game_data.get("home_away_record", "")
    away_home = game_data.get("away_home_record", "")
    away_away = game_data.get("away_away_record", "")

    home_line = f"- {home}: {home_rec}"
    if home_home:
        home_line += f" (home {home_home}, away {home_away})"
    lines.append(home_line)

    away_line = f"- {away}: {away_rec}"
    if away_home:
        away_line += f" (home {away_home}, away {away_away})"
    lines.append(away_line)

    # --- Recent Form ---
    lines.append("")
    lines.append("**Form**")
    if sport == "NBA":
        for team_name, prefix in [(home, "home"), (away, "away")]:
            form = game_data.get(f"{prefix}_form", {})
            if form.get("record"):
                lines.append(f"- {team_name}: L10 {form['record']}, {form.get('streak', '?')}")
    else:
        for team_name, prefix in [(home, "home"), (away, "away")]:
            l10 = game_data.get(f"{prefix}_l10", "")
            streak = game_data.get(f"{prefix}_streak", "")
            if l10:
                lines.append(f"- {team_name}: L10 {l10}, {streak}")

    # --- Key Players ---
    lines.append("")
    lines.append("**Key Players**")
    if sport == "NBA":
        for team_name, prefix in [(home, "home"), (away, "away")]:
            leaders = game_data.get(f"{prefix}_leaders", {})
            pts = leaders.get("pointspergame", leaders.get("points", {}))
            reb = leaders.get("reboundspergame", leaders.get("rebounds", {}))
            ast = leaders.get("assistspergame", leaders.get("assists", {}))
            rating = leaders.get("rating", {})

            if rating.get("name"):
                lines.append(f"- {team_name}: {rating['name']} ({rating['value']})")
            elif pts.get("name"):
                parts = [f"{pts['name']} {pts['value']} PPG"]
                if reb.get("name") and reb["name"] != pts["name"]:
                    parts.append(f"{reb['name']} {reb['value']} RPG")
                if ast.get("name") and ast["name"] != pts["name"]:
                    parts.append(f"{ast['name']} {ast['value']} APG")
                lines.append(f"- {team_name}: {', '.join(parts)}")
    else:
        # NHL advanced stats
        for team_name, prefix in [(home, "home"), (away, "away")]:
            adv = game_data.get(f"{prefix}_advanced", {})
            if adv:
                parts = []
                if adv.get("CF%"):
                    parts.append(f"CF% {adv['CF%']}")
                if adv.get("xGF%"):
                    parts.append(f"xGF% {adv['xGF%']}")
                if parts:
                    lines.append(f"- {team_name}: {', '.join(parts)}")

    # --- Goalies (NHL) ---
    if game_data.get("home_goalie") or game_data.get("away_goalie"):
        lines.append("")
        lines.append("**Goalies**")
        lines.append(f"- {home}: {game_data.get('home_goalie', 'TBD')}")
        lines.append(f"- {away}: {game_data.get('away_goalie', 'TBD')}")

    # --- Injuries ---
    lines.append("")
    lines.append("**Injuries**")
    injuries = game_data.get("injuries", [])
    notable = [i for i in injuries if (i.get("status") or "").lower() in ("out", "doubtful", "questionable", "day-to-day")]
    if notable:
        for inj in notable:
            status = (inj.get("status") or "unknown").upper()
            lines.append(f"- {inj.get('player', '?')} ({inj.get('team', '?')}): {status} — {inj.get('description', '')}")
    else:
        lines.append("- No notable absences reported")

    # --- Situational Factors ---
    if sit.factors:
        lines.append("")
        lines.append("**Situational**")
        for factor_desc in sit.factors.values():
            lines.append(f"- {factor_desc}")

    # --- Layers ---
    lines.append("")
    lines.append("**Layers**")
    # Show base with shrinkage note + split blend info
    base_line = f"- Base: {decomposition.base_probability:.1%} home win"
    if sport == "NBA":
        base_line += " (shrunk 0.70)"
    home_home = game_data.get("home_home_record", "")
    away_away = game_data.get("away_away_record", "")
    if home_home or away_away:
        base_line += f" | blend: 70% splits + 30% season"
    lines.append(base_line)
    if sit.total != 0:
        lines.append(f"- Situational: {sit.total:+.1%}")
    if info.total != 0:
        lines.append(f"- Information: {info.total:+.1%} — {info.reasoning}")
    else:
        lines.append(f"- Information: +0.0%")

    # --- Market + Edge ---
    market = game_data.get("market")
    if market:
        lines.append("")
        lines.append("**Market (Polymarket)**")
        lines.append(f"- {home}: {market['home_price']:.0%} / {away}: {market['away_price']:.0%}")
        lines.append(f"- Liquidity: ${market['liquidity']:,.0f}")

    if decomposition.edge:
        e = decomposition.edge
        lines.append("")
        lines.append("**Edge**")
        lines.append(f"- Model: {e.your_probability:.1%} vs Market: {e.true_implied:.1%}")
        lines.append(f"- **Effective edge: {e.effective_edge:+.1%}**")
        if e.passes_threshold:
            lines.append(f"- **BET** — passes {e.threshold_used:.0%} threshold, Kelly: {e.kelly_full:.1%} → {e.kelly_used:.1%} (1/4), bet: ${e.position_size:.2f}")
        else:
            lines.append(f"- **PASS** — below {e.threshold_used:.0%} threshold")

    # Model independence: gap between model and market before vig
    if market:
        market_implied = market.get("home_price", 0.5)
        model_gap = abs(home_prob - market_implied)
        independence_line = f"- Model independence: {model_gap:+.1%}"
        if model_gap < 0.03:
            independence_line += " — low model independence, tracks market"
        lines.append(independence_line)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Discord posting
# ---------------------------------------------------------------------------

def _format_game_time(iso_str: str) -> str:
    """Convert ISO datetime to readable ET time."""
    return format_game_time(iso_str, EASTERN_TZ, "ET")


def format_discord_report(
    game: dict,
    decomposition: ProbabilityDecomposition,
    summary: str,
) -> str:
    """Format research report for Discord: game header + deterministic summary."""
    home = game.get("home_team", "")
    away = game.get("away_team", "")
    tier = game.get("triage_level", "standard").replace("_", " ").title()
    game_time = _format_game_time(game.get("game_time", ""))

    home_prob = decomposition.final_probability
    away_prob = 1 - home_prob

    header = (
        f"**{away} @ {home}** — {game_time}\n"
        f"Triage: {tier}\n"
        f"Final: **{home} {home_prob:.1%}** / **{away} {away_prob:.1%}**\n"
    )

    return f"{header}\n{summary}"


def _load_discord_config() -> dict:
    config_path = Path(__file__).resolve().parent.parent.parent / ".discord_config.json"
    if not config_path.exists():
        return {}
    with open(config_path) as f:
        return json.load(f)


def _format_game_time_cst(iso_str: str) -> str:
    """Convert ISO datetime to CST for thread titles."""
    return format_game_time(iso_str, CENTRAL_TZ, "CT")


def post_to_discord(
    report: str,
    sport: str,
    game: dict | None = None,
) -> str | None:
    """Create a thread per game and post the report inside it.

    Returns the Discord thread ID on success, None on failure.
    """
    config = _load_discord_config()
    sport_key = sport.upper()
    if sport_key == "NBA":
        channel_key = "research_nba"
    elif sport_key == "NHL":
        channel_key = "research_nhl"
    elif sport_key == "TENNIS":
        channel_key = "research_tennis"
    else:
        channel_key = None
    channel_id = config.get("channels", {}).get(channel_key)
    bot_token = config.get("tokens", {}).get("research")

    if not channel_id or not bot_token:
        logger.error("Missing Discord config for %s", channel_key)
        return False

    headers = {
        "Authorization": f"Bot {bot_token}",
        "Content-Type": "application/json",
    }

    # Build thread title: "HOME_TEAM (HOME) vs AWAY_TEAM (AWAY) @ TIME CST"
    if game:
        if sport_key == "TENNIS":
            player_a = game.get("player_a") or game.get("home_team", "Unknown")
            player_b = game.get("player_b") or game.get("away_team", "Unknown")
            tournament = game.get("tournament", "")
            time_cst = _format_game_time_cst(game.get("scheduled_time") or game.get("game_time", ""))
            event_label = f" | {tournament}" if tournament else ""
            thread_title = f"{player_a} vs {player_b}{event_label} @ {time_cst}"
        else:
            home = game.get("home_team", "Unknown")
            away = game.get("away_team", "Unknown")
            time_cst = _format_game_time_cst(game.get("game_time", ""))
            thread_title = f"{home} (HOME) vs {away} (AWAY) @ {time_cst}"
    else:
        thread_title = f"Research — {sport}"

    if len(thread_title) > 100:
        thread_title = thread_title[:97] + "..."

    try:
        resp = requests.post(
            f"https://discord.com/api/v10/channels/{channel_id}/threads",
            headers=headers,
            json={
                "name": thread_title,
                "type": 11,
                "auto_archive_duration": 1440,
            },
            timeout=10,
        )
        if resp.status_code not in (200, 201):
            logger.error("Thread creation failed: %s %s", resp.status_code, resp.text)
            return None

        thread_id = resp.json().get("id")
    except Exception as e:
        logger.error("Thread creation error: %s", e)
        return None

    # Step 2: Post report inside the thread (split if needed)
    chunks = []
    if len(report) <= 2000:
        chunks = [report]
    else:
        # Split on double newlines
        parts = report.split("\n\n")
        current = ""
        for part in parts:
            if len(current) + len(part) + 2 > 1900 and current:
                chunks.append(current)
                current = part
            else:
                current += ("\n\n" if current else "") + part
        if current:
            chunks.append(current)
        # Hard-split safety
        final = []
        for chunk in chunks:
            while len(chunk) > 2000:
                final.append(chunk[:1990])
                chunk = chunk[1990:]
            if chunk:
                final.append(chunk)
        chunks = final

    for chunk in chunks:
        try:
            resp = requests.post(
                f"https://discord.com/api/v10/channels/{thread_id}/messages",
                headers=headers,
                json={"content": chunk},
                timeout=10,
            )
            if resp.status_code not in (200, 201):
                logger.error("Discord thread post failed: %s %s", resp.status_code, resp.text)
                return None
        except Exception as e:
            logger.error("Discord thread post error: %s", e)
            return None

    return thread_id


# ---------------------------------------------------------------------------
# Supabase storage
# ---------------------------------------------------------------------------

def store_to_supabase(
    game_id: int,
    decomposition: ProbabilityDecomposition,
    analysis: str | None,
    edge_type: str = "B",
    recommendation: str = "MONITOR",
    discord_thread_id: str | None = None,
    model_variant: str = "base",
) -> None:
    """Write research output to Supabase research table."""
    if not game_id:
        return
    _research_repo.store_prediction(
        game_id=game_id,
        decomposition=decomposition,
        edge_type=edge_type,
        recommendation=recommendation,
        discord_thread_id=discord_thread_id,
        model_variant=model_variant,
    )


# ---------------------------------------------------------------------------
# Trade signals — auto-post when edge passes threshold
# ---------------------------------------------------------------------------

def post_trade_signal(
    game: dict,
    decomposition: ProbabilityDecomposition,
    game_data: dict,
) -> bool:
    """Post to #trade-signals when edge passes threshold. Uses webhook embed."""
    # Determine which edge to display based on bet_side
    if decomposition.bet_side == "away" and decomposition.away_edge:
        e = decomposition.away_edge
    else:
        e = decomposition.edge
    if not e or not e.passes_threshold:
        return False

    home = decomposition.home_team
    away = decomposition.away_team
    home_prob = decomposition.final_probability
    away_prob = 1 - home_prob

    # Use bet_side when available (two-sided edge)
    if decomposition.bet_side == "away":
        pick, pick_prob = away, away_prob
    elif decomposition.bet_side == "home":
        pick, pick_prob = home, home_prob
    elif home_prob >= 0.5:
        pick, pick_prob = home, home_prob
    else:
        pick, pick_prob = away, away_prob

    game_time = _format_game_time(game.get("game_time", ""))
    market = game_data.get("market", {})

    embed = {
        "title": f"TRADE SIGNAL — {away} @ {home}",
        "description": f"**Pick: {pick} ({pick_prob:.0%})**",
        "color": 0x57F287,  # Green
        "fields": [
            {
                "name": "Edge",
                "value": (
                    f"Model: {e.your_probability:.1%} vs Market: {e.true_implied:.1%}\n"
                    f"**Effective: {e.effective_edge:+.1%}** (threshold: {e.threshold_used:.0%})"
                ),
                "inline": True,
            },
            {
                "name": "Position",
                "value": f"${e.position_size:.2f} ({e.position_pct:.1%})\nKelly: {e.kelly_full:.0%} → {e.kelly_used:.1%}",
                "inline": True,
            },
            {
                "name": "Market",
                "value": (
                    f"Liquidity: ${market.get('liquidity', 0):,.0f}\n"
                    f"Vig: {e.vig_estimate:.1%} / Spread: {abs(e.ask_price - e.bid_price):.1%}"
                ),
                "inline": False,
            },
        ],
        "footer": {"text": f"{game_time} — Polymarket moneyline"},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    config = _load_discord_config()
    sport_key = f"trade_signals_{game.get('sport', '').lower()}"

    # Webhook (preferred)
    webhook_url = (
        config.get("webhooks", {}).get(sport_key)
        or config.get("webhooks", {}).get("trade_signals")
    )
    if webhook_url:
        try:
            resp = requests.post(
                webhook_url,
                json={"username": "Trade Signal", "embeds": [embed]},
                timeout=10,
            )
            if resp.status_code in (200, 204):
                logger.info("Trade signal posted via webhook: %s", pick)
                return True
            else:
                logger.warning("Webhook failed: %s %s", resp.status_code, resp.text)
        except Exception as ex:
            logger.warning("Webhook error: %s", ex)

    # Fallback: bot API with embed
    channel_id = (
        config.get("channels", {}).get(sport_key)
        or config.get("channels", {}).get("trade_signals")
    )
    bot_token = config.get("tokens", {}).get("alert")
    if channel_id and bot_token:
        try:
            resp = requests.post(
                f"https://discord.com/api/v10/channels/{channel_id}/messages",
                headers={"Authorization": f"Bot {bot_token}", "Content-Type": "application/json"},
                json={"embeds": [embed]},
                timeout=10,
            )
            if resp.status_code in (200, 201):
                logger.info("Trade signal posted via bot: %s", pick)
                return True
        except Exception as ex:
            logger.error("Trade signal delivery failed: %s", ex)

    return False


# ---------------------------------------------------------------------------
# Main analysis pipeline
# ---------------------------------------------------------------------------

def analyse_game(game: dict, session_context: dict) -> dict | None:
    """Full analysis pipeline for a single game.

    Returns analysis result dict, or None for failures.
    """
    sport = game.get("sport", "").upper()
    home = game.get("home_team", "")
    away = game.get("away_team", "")
    tier = game.get("triage_level", "standard")
    game_id = game.get("id")

    logger.info("Analysing %s @ %s [%s] (%s)", away, home, tier, sport)

    artifacts = predict_game(game)
    game_data = dict(artifacts.snapshot.data)
    decomposition = artifacts.decomposition
    summary = artifacts.summary
    edge_type = artifacts.edge_type

    # 5. Post to Discord research thread
    report = format_discord_report(game, decomposition, summary)
    thread_id = post_to_discord(report, sport, game=game)

    # 6. Determine recommendation (trade signals posted by VPS executor after actual fill)
    signal_posted = False
    recommendation = artifacts.recommendation

    # 7. Store to Supabase (with thread ID for pre-game brief replies)
    if game_id:
        store_to_supabase(
            game_id,
            decomposition,
            summary,
            edge_type,
            recommendation,
            thread_id,
            model_variant=artifacts.model_variant,
        )

    prediction_record = build_prediction_record(None, artifacts)
    _snapshot_repo.append_snapshot({
        "snapshot": artifacts.snapshot,
        "feature_snapshot": artifacts.feature_snapshot,
        "model_variant": artifacts.model_variant,
    })
    _snapshot_repo.append_prediction_record(prediction_record)

    # 8. Safety layer — lineage logging + shadow testing
    try:
        from src.model.safety import get_lineage, get_shadow, PredictionAudit

        # Log prediction lineage
        home_rtg = game_data.get("home_net_rtg", {}).get("season", {})
        away_rtg = game_data.get("away_net_rtg", {}).get("season", {})
        audit = PredictionAudit(
            game_id=game_id or 0, sport=sport, home_team=home, away_team=away,
            timestamp=datetime.now(timezone.utc).isoformat(),
            net_rating_home=home_rtg.get("net_rtg", 0) if home_rtg else 0,
            net_rating_away=away_rtg.get("net_rtg", 0) if away_rtg else 0,
            players_out_home=[p.player_name for p in decomposition.information_edge.player_impacts if p.team == "home"],
            players_out_away=[p.player_name for p in decomposition.information_edge.player_impacts if p.team == "away"],
            market_price=game_data.get("market", {}).get("home_ask") if game_data.get("market") else None,
            base_prob=decomposition.base_probability,
            situational_adj=decomposition.situational_adjustment.total,
            info_edge=decomposition.information_edge.total,
            final_prob=decomposition.final_probability,
            effective_edge=decomposition.edge.effective_edge if decomposition.edge else 0,
        )
        get_lineage().log_prediction(audit)

        # Shadow testing: run old win%-Elo alongside new net rating (NBA only)
        if sport == "NBA" and game_data.get("home_net_rtg"):
            from src.model.baseline import nba_base_probability
            old_prob = nba_base_probability(
                game_data.get("home_win_pct", 0.5), game_data.get("away_win_pct", 0.5),
                home_home_pct=_parse_split_pct(game_data.get("home_home_record")),
                away_away_pct=_parse_split_pct(game_data.get("away_away_record")),
            )
            get_shadow().predict_both(
                game_id=game_id or 0, home_team=home, away_team=away,
                old_prob=old_prob, new_prob=decomposition.base_probability,
            )
    except Exception as e:
        logger.warning("Safety layer error (non-blocking): %s", e)

    # Pick the active edge (home or away, whichever bet_side points to)
    active_edge = decomposition.edge
    if decomposition.bet_side == "away" and decomposition.away_edge:
        active_edge = decomposition.away_edge

    return {
        "game_id": game_id,
        "sport": sport,
        "home_team": home,
        "away_team": away,
        "tier": tier,
        "final_probability": decomposition.final_probability,
        "recommendation": recommendation,
        "edge_type": edge_type,
        "posted": bool(thread_id),
        "signal_posted": signal_posted,
        "edge": round(active_edge.effective_edge, 4) if active_edge else None,
        "bet_side": decomposition.bet_side,
        "deterministic": True,
        "model_variant": artifacts.model_variant,
    }


def load_tennis_slate(match_date: date | None = None, tour: str | None = None) -> list[dict]:
    """Load a tennis slate from the provider-aware tennis client."""
    return tennis_schedule(match_date, tour=tour)


def format_tennis_discord_report(match: dict, artifacts) -> str:
    snapshot = artifacts.snapshot
    decomposition = artifacts.decomposition
    player_a_prob = decomposition.final_probability
    player_b_prob = 1.0 - player_a_prob
    time_ct = _format_game_time_cst(snapshot.scheduled_time)
    return (
        f"**{snapshot.tour} | {snapshot.player_a} vs {snapshot.player_b}**\n"
        f"{snapshot.tournament} | {snapshot.round_name} | {snapshot.surface.title()} | "
        f"Best of {snapshot.best_of} | {time_ct}\n\n"
        f"{artifacts.summary}\n\n"
        f"Final: **{snapshot.player_a} {player_a_prob:.1%}** / "
        f"**{snapshot.player_b} {player_b_prob:.1%}**"
    )


def analyse_tennis_match(match: dict, persist: bool = True) -> dict | None:
    """Run the tennis research path for a single match."""
    logger.info(
        "Analysing tennis match %s vs %s [%s]",
        match.get("player_a", ""),
        match.get("player_b", ""),
        match.get("tour", ""),
    )
    match_data = build_tennis_match_data(match)
    artifacts = predict_tennis(match, match_data)
    report = format_tennis_discord_report(match, artifacts)
    thread_id = post_to_discord(report, "TENNIS", game=match)
    if persist:
        persist_tennis_prediction(artifacts, repository=_snapshot_repo)

    return {
        "match_id": artifacts.snapshot.match_id,
        "sport": "TENNIS",
        "tour": artifacts.snapshot.tour,
        "player_a": artifacts.snapshot.player_a,
        "player_b": artifacts.snapshot.player_b,
        "tournament": artifacts.snapshot.tournament,
        "round_name": artifacts.snapshot.round_name,
        "final_probability_player_a": artifacts.decomposition.final_probability,
        "recommendation": artifacts.recommendation,
        "posted": bool(thread_id),
        "model_variant": artifacts.model_variant,
    }


def run_session():
    """Run full research session for today's slate.

    1. Load session context (lessons + prior grading)
    2. Load today's games from Supabase
    3. Analyse each game in triage order (deep first, then standard, then skip)
    4. Report summary
    """
    logger.info("Research Agent session starting for %s", date.today().isoformat())

    # Session start protocol
    context = load_session_context()
    if context.get("lessons"):
        logger.info("Loaded lessons.md (%d chars)", len(context["lessons"]))
    if context.get("prior_grading"):
        logger.info("Loaded %d prior calibration entries", len(context["prior_grading"]))

    # Load slate
    slate = load_todays_slate()
    if not slate:
        logger.warning("No games found in Supabase for today — running without game IDs")
        # Fallback: build slate from live data
        from src.scripts.morning_slate import build_nba_slate, build_nhl_slate
        nba = build_nba_slate()
        nhl = build_nhl_slate()
        slate = []
        for g in nba + nhl:
            slate.append({
                "id": None,
                "sport": g["sport"],
                "home_team": g["home_team"],
                "away_team": g["away_team"],
                "game_time": g["game_time"],
                "triage_level": g["triage"].lower().split()[0],
            })

    # Deduplicate games by (sport, home_team, away_team) — keep first occurrence
    seen = set()
    deduped = []
    for g in slate:
        key = (g.get("sport", ""), g.get("home_team", ""), g.get("away_team", ""))
        if key not in seen:
            seen.add(key)
            deduped.append(g)
    if len(deduped) < len(slate):
        logger.info("Deduplicated slate: %d -> %d games", len(slate), len(deduped))
    slate = deduped

    # Sort by game time (earliest first), then triage as tiebreaker
    triage_order = {"deep": 0, "standard": 1, "skip": 2}
    slate.sort(key=lambda g: (g.get("game_time", ""), triage_order.get(g.get("triage_level", "standard"), 1)))

    logger.info("Slate: %d games (%d deep, %d standard, %d skip)",
                len(slate),
                sum(1 for g in slate if g.get("triage_level") == "deep"),
                sum(1 for g in slate if g.get("triage_level") == "standard"),
                sum(1 for g in slate if g.get("triage_level") == "skip"))

    results = []
    for game in slate:
        try:
            result = analyse_game(game, context)
            if result:
                results.append(result)
                logger.info(
                    "  %s @ %s: %.1f%% home | %s | %s",
                    result["away_team"], result["home_team"],
                    result["final_probability"] * 100,
                    result["recommendation"],
                    "posted" if result["posted"] else "FAILED to post",
                )
        except Exception as e:
            logger.error("Failed to analyse %s @ %s: %s",
                         game.get("away_team"), game.get("home_team"), e,
                         exc_info=True)

    logger.info("Session complete: %d/%d games analysed", len(results), len(slate))

    # 8. Auto-execute BET signals via Polymarket
    bet_results = results  # full results list for return
    _auto_execute_trades(results, slate)

    return bet_results


def run_tennis_session(
    match_date: date | None = None,
    *,
    tour: str | None = None,
    limit: int | None = None,
    persist: bool = True,
) -> list[dict]:
    """Run a tennis research session from the provider-backed tennis slate."""
    target_date = match_date or date.today()
    logger.info("Tennis research session starting for %s [%s]", target_date.isoformat(), tour or "ALL")
    slate = load_tennis_slate(target_date, tour=tour)
    if limit is not None:
        slate = slate[:limit]
    if not slate:
        logger.warning("No tennis matches found for %s [%s]", target_date.isoformat(), tour or "ALL")
        return []

    results = []
    for match in slate:
        try:
            result = analyse_tennis_match(match, persist=persist)
            if result:
                results.append(result)
                logger.info(
                    "  %s vs %s: %.1f%% player_a | %s | %s",
                    result["player_a"],
                    result["player_b"],
                    result["final_probability_player_a"] * 100,
                    result["recommendation"],
                    "posted" if result["posted"] else "FAILED to post",
                )
        except Exception as e:
            logger.error(
                "Failed to analyse tennis match %s vs %s: %s",
                match.get("player_a"),
                match.get("player_b"),
                e,
                exc_info=True,
            )
    logger.info("Tennis session complete: %d/%d matches analysed", len(results), len(slate))
    return results


def _auto_execute_trades(results: list[dict], slate: list[dict]):
    """Execute trades for BET recommendations if trading is enabled.

    Strategy: execute only explicit BET signals that passed the model's
    thresholding and recommendation rules. Checks POLY_PRIVATE_KEY env var.
    """
    import os
    if not os.getenv("POLY_PRIVATE_KEY"):
        logger.info("Trading disabled (POLY_PRIVATE_KEY not set)")
        return

    tradeable = [
        r for r in results
        if r.get("recommendation") == "BET" and r.get("final_probability") is not None
    ]
    if not tradeable:
        logger.info("No BET signals to trade")
        return

    logger.info("Executing %d BET signal(s) via Polymarket", len(tradeable))

    try:
        from src.trading.executor import execute_bet_signals, Portfolio, get_live_bankroll

        bankroll = get_live_bankroll()
        portfolio = Portfolio(starting_balance=bankroll)

        # Build signal dicts for executor
        signals = []
        for r in tradeable:
            game_id = r.get("game_id")
            sport = r.get("sport", "")
            home = r.get("home_team", "")
            away = r.get("away_team", "")
            final_prob = r.get("final_probability", 0.5)

            # Determine which team to buy
            # Use bet_side from two-sided edge when available
            bet_side = r.get("bet_side")
            if bet_side == "away":
                team_to_buy = away
                model_prob = 1.0 - final_prob
            elif bet_side == "home":
                team_to_buy = home
                model_prob = final_prob
            elif final_prob >= 0.5:
                team_to_buy = home
                model_prob = final_prob
            else:
                team_to_buy = away
                model_prob = 1.0 - final_prob

            # Get live market data for execution
            market_data = get_market_prices(home, away, sport)
            if not market_data:
                logger.warning("No market data for %s @ %s — skipping trade", away, home)
                continue

            # Inject team names for token resolution
            market_data["_home_team"] = home
            market_data["_away_team"] = away
            market_data["_sport"] = sport

            # Calculate hours to game
            game_time_str = ""
            for g in slate:
                if g.get("id") == game_id:
                    game_time_str = g.get("game_time", "")
                    break
            hours_to_game = 6.0  # default
            if game_time_str:
                try:
                    gt = datetime.fromisoformat(game_time_str.replace("Z", "+00:00"))
                    hours_to_game = max(0.5, (gt - datetime.now(timezone.utc)).total_seconds() / 3600)
                except (ValueError, TypeError):
                    pass

            signals.append({
                "game_id": game_id,
                "sport": sport,
                "home_team": home,
                "away_team": away,
                "team_to_buy": team_to_buy,
                "model_prob": model_prob,
                "market_data": market_data,
                "hours_to_game": hours_to_game,
                "edge_type": r.get("edge_type", "B"),
                "base_prob": 0.0,
                "situational_adj": 0.0,
                "info_edge": 0.0,
            })

        if signals:
            trade_results = execute_bet_signals(signals, portfolio)
            executed = sum(1 for t in trade_results if t.success)
            logger.info("Trading complete: %d/%d executed", executed, len(signals))
        else:
            logger.info("No valid signals with market data")

    except Exception as e:
        logger.error("Auto-execution failed: %s", e, exc_info=True)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if len(sys.argv) > 1 and sys.argv[1] == "single":
        # Analyse a single game by index
        context = load_session_context()
        slate = load_todays_slate()
        if not slate:
            from src.scripts.morning_slate import build_nba_slate, build_nhl_slate
            nba = build_nba_slate()
            nhl = build_nhl_slate()
            slate = [{"id": None, "sport": g["sport"], "home_team": g["home_team"],
                       "away_team": g["away_team"], "game_time": g["game_time"],
                       "triage_level": g["triage"].lower().split()[0]} for g in nba + nhl]

        idx = int(sys.argv[2]) if len(sys.argv) > 2 else 0
        if idx < len(slate):
            result = analyse_game(slate[idx], context)
            print(json.dumps(result, indent=2))
    elif len(sys.argv) > 1 and sys.argv[1] == "tennis":
        tour = sys.argv[2] if len(sys.argv) > 2 else None
        results = run_tennis_session(tour=tour)
        print(json.dumps(results, indent=2))
    else:
        run_session()
