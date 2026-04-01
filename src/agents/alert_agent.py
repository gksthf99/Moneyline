"""
Alert Agent — pre-game briefs with probability impact.

Fires ONE comprehensive alert per game ~30 minutes before start.
Includes: confirmed goalies (with save%, starter/backup), key injuries
(with player tier and impact), and net probability adjustment.

No intermediate alerts — just the final pre-game picture.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta, timezone
from pathlib import Path

import requests

from src.config import SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY
from src.data.nba_espn import get_injuries as nba_injuries
from src.data.nhl_dailyfaceoff import get_starting_goalies
from src.data.nhl_goalies import get_team_goalies, identify_goalie, goalie_adjustment
from src.time_utils import CENTRAL_TZ, current_sports_date, format_game_time, sports_day_bounds

logger = logging.getLogger(__name__)

SUPABASE_HEADERS = {
    "apikey": SUPABASE_SERVICE_ROLE_KEY or "",
    "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY or ''}",
    "Content-Type": "application/json",
}

# NHL team name → abbreviation mapping (for goalie stat lookups)
_TEAM_ABBREVS = {
    "Anaheim Ducks": "ANA", "Boston Bruins": "BOS", "Buffalo Sabres": "BUF",
    "Calgary Flames": "CGY", "Carolina Hurricanes": "CAR", "Chicago Blackhawks": "CHI",
    "Colorado Avalanche": "COL", "Columbus Blue Jackets": "CBJ", "Dallas Stars": "DAL",
    "Detroit Red Wings": "DET", "Edmonton Oilers": "EDM", "Florida Panthers": "FLA",
    "Los Angeles Kings": "LAK", "Minnesota Wild": "MIN", "Montreal Canadiens": "MTL",
    "Montréal Canadiens": "MTL",
    "Nashville Predators": "NSH", "New Jersey Devils": "NJD", "New York Islanders": "NYI",
    "New York Rangers": "NYR", "Ottawa Senators": "OTT", "Philadelphia Flyers": "PHI",
    "Pittsburgh Penguins": "PIT", "San Jose Sharks": "SJS", "Seattle Kraken": "SEA",
    "St Louis Blues": "STL", "St. Louis Blues": "STL",
    "Tampa Bay Lightning": "TBL", "Toronto Maple Leafs": "TOR",
    "Utah Mammoth": "UTA", "Utah Hockey Club": "UTA",
    "Vancouver Canucks": "VAN", "Vegas Golden Knights": "VGK",
    "Winnipeg Jets": "WPG", "Washington Capitals": "WSH",
}


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class PreGameBrief:
    """One comprehensive pre-game alert per game."""
    game_id: int | None
    sport: str
    home_team: str
    away_team: str
    game_time: str  # ISO
    game_time_display: str  # human-readable

    # NHL goalie info
    home_goalie: str | None = None
    away_goalie: str | None = None
    home_goalie_status: str = "Unconfirmed"
    away_goalie_status: str = "Unconfirmed"
    home_goalie_detail: str = ""  # "Starter, .912 SV%, 25W-10L"
    away_goalie_detail: str = ""
    goalie_impact: float = 0.0  # probability adjustment from goalie matchup
    goalie_impact_desc: str = ""

    # Injury info
    injuries: list[dict] = field(default_factory=list)  # [{player, team, status, tier, impact}]
    injury_impact: float = 0.0
    injury_impact_desc: str = ""

    # Model context
    model_prob: float | None = None  # original morning probability
    adjusted_prob: float | None = None  # repriced pre-game probability
    recommendation: str = ""  # revised recommendation
    original_recommendation: str = ""
    original_effective_edge: float | None = None
    revised_effective_edge: float | None = None
    repriced: bool = False

    # Discord thread for reply
    discord_thread_id: str | None = None

    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# State — tracks which games have already been alerted
# ---------------------------------------------------------------------------

class AlertState:
    def __init__(self):
        self.alerted_games: set[int] = set()  # game IDs that got a pre-game brief

    def was_alerted(self, game_id: int) -> bool:
        return game_id in self.alerted_games

    def mark_alerted(self, game_id: int) -> None:
        self.alerted_games.add(game_id)


_state = AlertState()


def hydrate_state() -> None:
    """Load today's already-fired alerts from Supabase to avoid duplicates on restart."""
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        return
    lower, upper = sports_day_bounds()
    try:
        resp = requests.get(
            f"{SUPABASE_URL}/rest/v1/alerts",
            headers=SUPABASE_HEADERS,
            params={
                "and": f"(created_at.gte.{lower},created_at.lt.{upper})",
                "alert_type": "eq.pre_game_brief",
                "select": "game_id",
            },
            timeout=10,
        )
        if resp.status_code == 200:
            for row in resp.json():
                gid = row.get("game_id")
                if gid:
                    _state.alerted_games.add(gid)
            logger.info("Hydrated state: %d games already alerted today", len(_state.alerted_games))
    except Exception as e:
        logger.warning("State hydration failed: %s", e)


# ---------------------------------------------------------------------------
# Build pre-game brief for a single game
# ---------------------------------------------------------------------------

def _format_game_time(iso_str: str) -> str:
    """Convert ISO datetime to readable CT time."""
    return format_game_time(iso_str, CENTRAL_TZ, "CT")


def _goalie_detail_str(goalie_name: str | None, stats: dict | None, team_abbrev: str) -> str:
    """Build a one-line goalie description: 'Starter, .912 SV%, 25W-10L-3OTL'."""
    if not goalie_name:
        return "TBD"

    # Try to identify from NHL API stats (has starter/backup detection)
    nhl_goalie = identify_goalie(goalie_name, team_abbrev)

    # Prefer NHL API stats (more authoritative), fall back to Daily Faceoff stats
    if nhl_goalie:
        role = "Starter" if nhl_goalie.get("is_starter") else "Backup"
        sv = nhl_goalie.get("save_pct", 0)
        w = nhl_goalie.get("wins", 0)
        l = nhl_goalie.get("losses", 0)
        otl = nhl_goalie.get("ot_losses", 0)
        return f"{role}, .{int(sv*1000):03d} SV%, {w}W-{l}L-{otl}OTL"
    elif stats and stats.get("save_pct"):
        sv = stats["save_pct"]
        w = stats.get("wins", 0) or 0
        l = stats.get("losses", 0) or 0
        otl = stats.get("otl", 0) or 0
        return f".{int(sv*1000):03d} SV%, {w}W-{l}L-{otl}OTL"

    return "No stats available"


def _get_nba_injury_tier(ppg: float) -> tuple[str, float]:
    """Classify NBA player by PPG into impact tier. Returns (tier_label, max_impact%)."""
    if ppg >= 20:
        return "T1 Star", 0.06
    elif ppg >= 15:
        return "T2 Key", 0.035
    elif ppg >= 10:
        return "T3 Rotation", 0.02
    else:
        return "T4 Role", 0.0075


def _get_research_for_game(game_id: int) -> dict | None:
    """Get latest research prediction for a game (including Discord thread ID)."""
    if not SUPABASE_URL:
        return None
    try:
        resp = requests.get(
            f"{SUPABASE_URL}/rest/v1/research",
            headers=SUPABASE_HEADERS,
            params={
                "game_id": f"eq.{game_id}",
                "select": "id,prob_decomposition,recommendation,effective_edge,edge_type,discord_thread_id",
                "order": "id.desc",
                "limit": "1",
            },
            timeout=10,
        )
        if resp.status_code == 200 and resp.json():
            return resp.json()[0]
    except Exception:
        pass
    return None


def _edge_type_for_decomposition(decomposition) -> str:
    """Mirror the research agent's edge type classification."""
    return "C" if decomposition.information_edge.total != 0 else "B"


def _recommendation_for_decomposition(decomposition) -> str:
    """Mirror the research agent's recommendation logic."""
    if decomposition.edge and decomposition.edge.passes_threshold:
        return "BET"
    if decomposition.edge and not decomposition.edge.passes_threshold:
        return "PASS"
    return "MONITOR"


def _persist_repriced_research(game: dict, brief: PreGameBrief, decomposition) -> None:
    """Append a new research row representing the pre-game repricing."""
    if not game.get("id"):
        return

    try:
        from src.agents.research_agent import store_to_supabase
    except Exception as e:
        logger.warning("Unable to import research storage helper: %s", e)
        return

    old_prob = brief.model_prob
    new_prob = brief.adjusted_prob
    if old_prob is not None and new_prob is not None:
        analysis = (
            f"Pre-game repricing due to {'goalie update' if brief.sport == 'NHL' else 'injury update'}. "
            f"Morning {old_prob:.1%} -> Pregame {new_prob:.1%}. "
            f"Signal {brief.original_recommendation or 'N/A'} -> {brief.recommendation}."
        )
    else:
        analysis = "Pre-game repricing from alert agent."

    store_to_supabase(
        game_id=game["id"],
        decomposition=decomposition,
        analysis=analysis,
        edge_type=_edge_type_for_decomposition(decomposition),
        recommendation=brief.recommendation,
        discord_thread_id=brief.discord_thread_id,
        model_variant="base",
    )


def _reprice_brief(game: dict, brief: PreGameBrief, research: dict | None) -> None:
    """Recompute the game's probability from current pre-game data and persist it."""
    if not research:
        return

    decomp = research.get("prob_decomposition", {})
    if isinstance(decomp, str):
        try:
            decomp = json.loads(decomp)
        except json.JSONDecodeError:
            decomp = {}

    brief.model_prob = decomp.get("final")
    brief.original_recommendation = research.get("recommendation", "") or ""
    brief.original_effective_edge = research.get("effective_edge")
    brief.discord_thread_id = research.get("discord_thread_id")

    try:
        from src.services.game_prediction import predict_game

        repriced = predict_game(game).decomposition
    except Exception as e:
        logger.warning("Failed to reprice %s @ %s: %s", brief.away_team, brief.home_team, e)
        return

    brief.adjusted_prob = repriced.final_probability
    brief.recommendation = _recommendation_for_decomposition(repriced)
    brief.revised_effective_edge = repriced.edge.effective_edge if repriced.edge else None
    brief.repriced = True

    _persist_repriced_research(game, brief, repriced)


def build_nhl_brief(game: dict, goalie_matchups: list[dict]) -> PreGameBrief | None:
    """Build pre-game brief for an NHL game."""
    home = game["home_team"]
    away = game["away_team"]
    game_time = game.get("game_time", "")

    brief = PreGameBrief(
        game_id=game["id"],
        sport="NHL",
        home_team=home,
        away_team=away,
        game_time=game_time,
        game_time_display=_format_game_time(game_time),
    )

    # Find goalie matchup for this game
    home_lower = home.lower()
    away_lower = away.lower()
    matchup = None
    for m in goalie_matchups:
        if (m["home_team"].lower() in home_lower or home_lower in m["home_team"].lower()
                or m["away_team"].lower() in away_lower or away_lower in m["away_team"].lower()):
            matchup = m
            break

    # Also try matching by team slug keywords
    if not matchup:
        for m in goalie_matchups:
            # Match by last word of team name (mascot)
            home_mascot = home.split()[-1].lower()
            away_mascot = away.split()[-1].lower()
            if (home_mascot in m["home_team"].lower() or home_mascot in m["away_team"].lower()
                    or away_mascot in m["home_team"].lower() or away_mascot in m["away_team"].lower()):
                matchup = m
                break

    if matchup:
        home_abbrev = _TEAM_ABBREVS.get(home, "")
        away_abbrev = _TEAM_ABBREVS.get(away, "")

        brief.home_goalie = matchup.get("home_goalie")
        brief.away_goalie = matchup.get("away_goalie")
        brief.home_goalie_status = matchup.get("home_status", "Unconfirmed")
        brief.away_goalie_status = matchup.get("away_status", "Unconfirmed")
        brief.home_goalie_detail = _goalie_detail_str(
            brief.home_goalie, matchup.get("home_goalie_stats"), home_abbrev
        )
        brief.away_goalie_detail = _goalie_detail_str(
            brief.away_goalie, matchup.get("away_goalie_stats"), away_abbrev
        )

        # Calculate probability impact from goalie matchup
        if home_abbrev and away_abbrev:
            adj, desc = goalie_adjustment(
                brief.home_goalie, brief.away_goalie, home_abbrev, away_abbrev
            )
            brief.goalie_impact = adj
            brief.goalie_impact_desc = desc

    # Get model prediction + thread ID, then reprice from current pre-game state
    research = _get_research_for_game(game["id"])
    if research:
        _reprice_brief(game, brief, research)

    return brief


def build_nba_brief(game: dict, injury_data: list[dict]) -> PreGameBrief | None:
    """Build pre-game brief for an NBA game."""
    home = game["home_team"]
    away = game["away_team"]
    game_time = game.get("game_time", "")

    brief = PreGameBrief(
        game_id=game["id"],
        sport="NBA",
        home_team=home,
        away_team=away,
        game_time=game_time,
        game_time_display=_format_game_time(game_time),
    )

    # Filter injuries for this game's teams
    home_lower = home.lower()
    away_lower = away.lower()

    for inj in injury_data:
        team = (inj.get("team") or "").lower()
        status = (inj.get("status") or "").lower()
        player = inj.get("player", "")
        ppg = float(inj.get("ppg", 0) or 0)

        if team not in home_lower and home_lower not in team and team not in away_lower and away_lower not in team:
            continue

        # Only report meaningful statuses
        if status not in ("out", "doubtful", "questionable", "day-to-day"):
            continue

        tier_label, max_impact = _get_nba_injury_tier(ppg)

        # Skip T4 role players unless they're OUT
        if tier_label == "T4 Role" and status not in ("out",):
            continue

        brief.injuries.append({
            "player": player,
            "team": inj.get("team", ""),
            "status": status.upper(),
            "tier": tier_label,
            "ppg": ppg,
            "description": inj.get("description", ""),
        })

    # Get model prediction + thread ID, then reprice from current pre-game state
    research = _get_research_for_game(game["id"])
    if research:
        _reprice_brief(game, brief, research)

    return brief


# ---------------------------------------------------------------------------
# Discord delivery
# ---------------------------------------------------------------------------

def _load_discord_config() -> dict:
    config_path = Path(__file__).resolve().parent.parent.parent / ".discord_config.json"
    if not config_path.exists():
        return {}
    with open(config_path) as f:
        return json.load(f)


def _sport_channel_key(prefix: str, sport: str) -> str:
    return f"{prefix}_{sport.lower()}"


def _format_brief_message(brief: PreGameBrief) -> str:
    """Format pre-game brief as a Discord message."""
    lines = []
    lines.append(f"**PRE-GAME BRIEF** — {brief.away_team} @ {brief.home_team} — {brief.game_time_display}")
    lines.append("")

    # Goalies (NHL)
    if brief.sport == "NHL":
        lines.append("**Goalies:**")
        away_icon = {"Confirmed": "[C]", "Likely": "[L]", "Unconfirmed": "[?]"}.get(brief.away_goalie_status, "[?]")
        home_icon = {"Confirmed": "[C]", "Likely": "[L]", "Unconfirmed": "[?]"}.get(brief.home_goalie_status, "[?]")
        lines.append(f"  Away: {brief.away_goalie or 'TBD'} {away_icon} — {brief.away_goalie_detail}")
        lines.append(f"  Home: {brief.home_goalie or 'TBD'} {home_icon} — {brief.home_goalie_detail}")
        if brief.goalie_impact_desc:
            sign = "+" if brief.goalie_impact >= 0 else ""
            lines.append(f"  **Impact:** {sign}{brief.goalie_impact:.1%} home win prob — {brief.goalie_impact_desc}")
        lines.append("")

    # Injuries (NBA primarily, but also NHL if added later)
    if brief.injuries:
        lines.append("**Key Injuries:**")
        for inj in sorted(brief.injuries, key=lambda x: x.get("ppg", 0), reverse=True):
            ppg_str = f"{inj['ppg']:.1f} PPG" if inj.get("ppg") else ""
            lines.append(f"  {inj['player']} ({inj['team']}) — **{inj['status']}** [{inj['tier']}] {ppg_str}")
            if inj.get("description"):
                lines.append(f"    _{inj['description']}_")
        lines.append("")

    # Model repricing
    if brief.model_prob is not None:
        home_pct = brief.model_prob * 100
        away_pct = (1 - brief.model_prob) * 100
        lines.append(f"**Morning:** {brief.home_team} {home_pct:.0f}% / {brief.away_team} {away_pct:.0f}%")
    if brief.adjusted_prob is not None:
        adj_home = brief.adjusted_prob * 100
        adj_away = (1 - brief.adjusted_prob) * 100
        lines.append(f"**Pregame:** {brief.home_team} {adj_home:.0f}% / {brief.away_team} {adj_away:.0f}%")
        if brief.model_prob is not None:
            lines.append(f"**Delta:** {brief.adjusted_prob - brief.model_prob:+.1%} home win probability")
    if brief.original_recommendation and brief.recommendation:
        lines.append(f"**Signal:** {brief.original_recommendation} -> {brief.recommendation}")
    elif brief.recommendation:
        lines.append(f"**Signal:** {brief.recommendation}")

    return "\n".join(lines)


def _build_brief_embed(brief: PreGameBrief) -> dict:
    """Build a Discord embed for #alerts with trade-signals-style UI."""
    fields = []

    if brief.sport == "NHL":
        away_icon = {"Confirmed": "[C]", "Likely": "[L]", "Unconfirmed": "[?]"}.get(brief.away_goalie_status, "[?]")
        home_icon = {"Confirmed": "[C]", "Likely": "[L]", "Unconfirmed": "[?]"}.get(brief.home_goalie_status, "[?]")
        goalie_lines = [
            f"{brief.away_team}: {brief.away_goalie or 'TBD'} {away_icon}",
            f"{brief.home_team}: {brief.home_goalie or 'TBD'} {home_icon}",
        ]
        if brief.goalie_impact_desc:
            sign = "+" if brief.goalie_impact >= 0 else ""
            goalie_lines.append(f"Impact: {sign}{brief.goalie_impact:.1%}")
            goalie_lines.append(brief.goalie_impact_desc)
        fields.append({
            "name": "Goalies",
            "value": "\n".join(goalie_lines)[:1024],
            "inline": False,
        })

    if brief.injuries:
        injury_lines = []
        for inj in sorted(brief.injuries, key=lambda x: x.get("ppg", 0), reverse=True)[:8]:
            ppg_str = f" {inj['ppg']:.1f} PPG" if inj.get("ppg") else ""
            injury_lines.append(
                f"{inj['player']} ({inj['team']}) - {inj['status']} [{inj['tier']}]"
                f"{ppg_str}"
            )
        fields.append({
            "name": "Key Injuries",
            "value": "\n".join(injury_lines)[:1024],
            "inline": False,
        })

    if brief.model_prob is not None or brief.adjusted_prob is not None:
        model_lines = []
        if brief.model_prob is not None:
            home_pct = brief.model_prob * 100
            away_pct = (1 - brief.model_prob) * 100
            model_lines.append(f"Morning: {brief.home_team} {home_pct:.0f}% / {brief.away_team} {away_pct:.0f}%")
        if brief.adjusted_prob is not None:
            adj_home = brief.adjusted_prob * 100
            adj_away = (1 - brief.adjusted_prob) * 100
            model_lines.append(f"Pregame: {brief.home_team} {adj_home:.0f}% / {brief.away_team} {adj_away:.0f}%")
        if brief.adjusted_prob is not None and brief.model_prob is not None:
            model_lines.append(f"Delta: {brief.adjusted_prob - brief.model_prob:+.1%}")
        if brief.original_recommendation and brief.recommendation:
            model_lines.append(f"Signal: {brief.original_recommendation} -> {brief.recommendation}")
        elif brief.recommendation:
            model_lines.append(f"Signal: {brief.recommendation}")
        fields.append({
            "name": "Model",
            "value": "\n".join(model_lines)[:1024],
            "inline": False,
        })

    return {
        "title": f"PRE-GAME BRIEF - {brief.away_team} @ {brief.home_team}",
        "description": f"**{brief.game_time_display}**",
        "color": 0x57F287,
        "fields": fields,
        "footer": {"text": f"{brief.sport} pre-game update"},
        "timestamp": brief.created_at.isoformat(),
    }


def _format_thread_update(brief: PreGameBrief) -> str:
    """Format a shorter update message for replying to the research thread."""
    lines = []
    lines.append(f"**PRE-GAME UPDATE** — {brief.game_time_display}")
    lines.append("")

    if brief.sport == "NHL" and (brief.home_goalie or brief.away_goalie):
        lines.append("**Goalies:**")
        if brief.away_goalie:
            icon = {"Confirmed": "[C]", "Likely": "[L]"}.get(brief.away_goalie_status, "[?]")
            lines.append(f"  Away: {brief.away_goalie} {icon} — {brief.away_goalie_detail}")
        if brief.home_goalie:
            icon = {"Confirmed": "[C]", "Likely": "[L]"}.get(brief.home_goalie_status, "[?]")
            lines.append(f"  Home: {brief.home_goalie} {icon} — {brief.home_goalie_detail}")
        if brief.goalie_impact_desc:
            sign = "+" if brief.goalie_impact >= 0 else ""
            lines.append(f"  **Impact:** {sign}{brief.goalie_impact:.1%} — {brief.goalie_impact_desc}")
        lines.append("")

    if brief.injuries:
        lines.append("**Injuries:**")
        for inj in sorted(brief.injuries, key=lambda x: x.get("ppg", 0), reverse=True):
            lines.append(f"  {inj['player']} — **{inj['status']}** [{inj['tier']}] {inj.get('ppg', 0):.1f} PPG")
        lines.append("")

    if brief.adjusted_prob is not None and brief.model_prob is not None:
        home_orig = brief.model_prob * 100
        home_adj = brief.adjusted_prob * 100
        lines.append(f"**Probability:** {brief.home_team} {home_orig:.0f}% -> **{home_adj:.0f}%**")
        lines.append(f"**Delta:** {brief.adjusted_prob - brief.model_prob:+.1%} home win probability")
    elif brief.adjusted_prob is not None:
        lines.append(f"**Pregame Probability:** {brief.home_team} {brief.adjusted_prob:.0%}")

    if brief.original_recommendation and brief.recommendation:
        lines.append(f"**Signal:** {brief.original_recommendation} -> {brief.recommendation}")
    elif brief.recommendation:
        lines.append(f"**Signal:** {brief.recommendation}")

    return "\n".join(lines)


def reply_to_research_thread(brief: PreGameBrief) -> bool:
    """Post an update reply to the game's research thread in #research-[sport]."""
    if not brief.discord_thread_id:
        return False

    config = _load_discord_config()
    bot_token = config.get("tokens", {}).get("research")
    if not bot_token:
        logger.warning("No research bot token for thread reply")
        return False

    message = _format_thread_update(brief)

    try:
        resp = requests.post(
            f"https://discord.com/api/v10/channels/{brief.discord_thread_id}/messages",
            headers={
                "Authorization": f"Bot {bot_token}",
                "Content-Type": "application/json",
            },
            json={"content": message},
            timeout=10,
        )
        if resp.status_code in (200, 201):
            logger.info("Thread update posted: %s @ %s", brief.away_team, brief.home_team)
            return True
        else:
            logger.warning("Thread reply failed: %s %s", resp.status_code, resp.text)
            return False
    except Exception as e:
        logger.warning("Thread reply error: %s", e)
        return False


def _is_valid_webhook_url(url: str | None) -> bool:
    """Return True only for real Discord webhook URLs, not placeholders."""
    if not url:
        return False
    return url.startswith("https://discord.com/api/webhooks/")


def deliver_brief(brief: PreGameBrief) -> tuple[bool, bool, str]:
    """Post pre-game brief to #alerts via webhook, fallback to bot API.

    Returns:
        (delivered, fallback_triggered, delivery_channel)
    """
    config = _load_discord_config()
    message = _format_brief_message(brief)
    embed = _build_brief_embed(brief)

    channel_name = _sport_channel_key("alerts", brief.sport)
    legacy_channel_name = "alerts"

    # Try webhook first
    webhook_url = (
        config.get("webhooks", {}).get(channel_name)
        or config.get("webhooks", {}).get(legacy_channel_name)
    )
    if _is_valid_webhook_url(webhook_url):
        try:
            resp = requests.post(
                webhook_url,
                json={"username": "Pre-Game Alert", "embeds": [embed]},
                timeout=10,
            )
            if resp.status_code in (200, 204):
                logger.info("Brief delivered via webhook to #%s: %s @ %s",
                            channel_name, brief.away_team, brief.home_team)
                return True, False, f"discord:#{channel_name}:webhook"
            else:
                logger.warning("Webhook failed (%s), falling back to bot API", resp.status_code)
        except Exception as e:
            logger.warning("Webhook error (%s), falling back to bot API", e)
    else:
        logger.warning("Alerts webhook missing or placeholder; using bot delivery for #alerts")

    # Fallback: bot API
    channel_id = (
        config.get("channels", {}).get(channel_name)
        or config.get("channels", {}).get(legacy_channel_name)
    )
    bot_token = config.get("tokens", {}).get("alert")

    if not channel_id or not bot_token:
        logger.error("Missing Discord config for channel=%s", channel_name)
        return False, False, f"discord:#{channel_name}"

    try:
        resp = requests.post(
            f"https://discord.com/api/v10/channels/{channel_id}/messages",
            headers={
                "Authorization": f"Bot {bot_token}",
                "Content-Type": "application/json",
            },
            json={"content": message, "embeds": [embed]},
            timeout=10,
        )
        if resp.status_code in (200, 201):
            logger.info("Brief delivered via bot to #%s: %s @ %s",
                        channel_name, brief.away_team, brief.home_team)
            return True, True, f"discord:#{channel_name}:bot"
        else:
            logger.error("Discord delivery failed: %s %s", resp.status_code, resp.text)
            return False, True, f"discord:#{channel_name}:bot"
    except Exception as e:
        logger.error("Discord delivery error: %s", e)
        return False, True, f"discord:#{channel_name}:bot"


# ---------------------------------------------------------------------------
# Supabase logging
# ---------------------------------------------------------------------------

def log_brief_to_supabase(
    brief: PreGameBrief,
    delivered: bool,
    fallback_triggered: bool,
    delivery_channel: str,
) -> None:
    """Log pre-game brief to alerts table."""
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        return

    content = {
        "sport": brief.sport,
        "home_goalie": brief.home_goalie,
        "away_goalie": brief.away_goalie,
        "home_goalie_status": brief.home_goalie_status,
        "away_goalie_status": brief.away_goalie_status,
        "home_goalie_detail": brief.home_goalie_detail,
        "away_goalie_detail": brief.away_goalie_detail,
        "goalie_impact": brief.goalie_impact,
        "goalie_impact_desc": brief.goalie_impact_desc,
        "injuries": brief.injuries,
        "model_prob": brief.model_prob,
        "adjusted_prob": brief.adjusted_prob,
        "recommendation": brief.recommendation,
    }

    try:
        requests.post(
            f"{SUPABASE_URL}/rest/v1/alerts",
            headers={**SUPABASE_HEADERS, "Prefer": "return=minimal"},
            json={
                "game_id": brief.game_id,
                "alert_type": "pre_game_brief",
                "source": "alert_agent",
                "content": json.dumps(content),
                "delivery_channel": delivery_channel,
                "delivered": delivered,
                "fallback_triggered": fallback_triggered,
            },
            timeout=10,
        )
    except Exception as e:
        logger.warning("Failed to log brief to Supabase: %s", e)


# ---------------------------------------------------------------------------
# Main cycle — check for games approaching, fire briefs
# ---------------------------------------------------------------------------

PRE_GAME_WINDOW_MINUTES = 30  # fire brief this many minutes before game


def _get_todays_games() -> list[dict]:
    """Get all games for today from Supabase."""
    if not SUPABASE_URL:
        return []

    lower, upper = sports_day_bounds()

    try:
        resp = requests.get(
            f"{SUPABASE_URL}/rest/v1/games",
            headers=SUPABASE_HEADERS,
            params={
                "and": f"(game_time.gte.{lower},game_time.lt.{upper})",
                "select": "id,sport,home_team,away_team,game_time",
                "order": "game_time",
            },
            timeout=10,
        )
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        logger.warning("Failed to fetch games: %s", e)
    return []


def run_cycle() -> list[PreGameBrief]:
    """Run one cycle: check for games within the pre-game window, fire briefs.

    Returns list of briefs that were delivered.
    """
    now = datetime.now(timezone.utc)
    games = _get_todays_games()

    if not games:
        return []

    # Find games within the pre-game window that haven't been alerted
    games_to_brief = []
    for g in games:
        gid = g["id"]
        if _state.was_alerted(gid):
            continue

        gt_str = g.get("game_time", "")
        if not gt_str:
            continue

        try:
            gt = datetime.fromisoformat(gt_str.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            continue

        minutes_until = (gt - now).total_seconds() / 60

        # Fire brief if game is within window (and hasn't started yet)
        if 0 <= minutes_until <= PRE_GAME_WINDOW_MINUTES:
            games_to_brief.append(g)

    if not games_to_brief:
        return []

    # Fetch goalie matchups and injury data once for all games
    goalie_matchups = []
    injury_data = []

    nhl_games = [g for g in games_to_brief if g["sport"] == "NHL"]
    nba_games = [g for g in games_to_brief if g["sport"] == "NBA"]

    if nhl_games:
        try:
            goalie_matchups = get_starting_goalies(current_sports_date())
        except Exception as e:
            logger.warning("Failed to fetch goalie starts: %s", e)

    if nba_games:
        try:
            injury_data = nba_injuries()
        except Exception as e:
            logger.warning("Failed to fetch NBA injuries: %s", e)

    # Build and deliver briefs
    delivered = []
    for g in games_to_brief:
        if g["sport"] == "NHL":
            brief = build_nhl_brief(g, goalie_matchups)
        elif g["sport"] == "NBA":
            brief = build_nba_brief(g, injury_data)
        else:
            continue

        if not brief:
            continue

        success, fallback_triggered, delivery_channel = deliver_brief(brief)
        log_brief_to_supabase(brief, success, fallback_triggered, delivery_channel)

        # Also reply to the research thread with the update
        reply_to_research_thread(brief)

        if success:
            _state.mark_alerted(brief.game_id)
            delivered.append(brief)
            logger.info("Pre-game brief fired: %s @ %s (%s)", brief.away_team, brief.home_team, brief.sport)

    return delivered


# ---------------------------------------------------------------------------
# Daemon mode
# ---------------------------------------------------------------------------

def run_daemon(check_interval: int = 300):
    """Run alert daemon. Checks every 5 minutes for games entering the pre-game window.

    Args:
        check_interval: Seconds between checks (default: 5 min).
    """
    import time

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger.info("Alert Agent daemon starting (pre-game brief mode, check every %ds)", check_interval)

    hydrate_state()

    while True:
        try:
            briefs = run_cycle()
            if briefs:
                logger.info("Delivered %d pre-game brief(s)", len(briefs))
        except Exception as e:
            logger.error("Cycle failed: %s", e, exc_info=True)

        time.sleep(check_interval)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "daemon":
        interval = int(sys.argv[2]) if len(sys.argv) > 2 else 300
        run_daemon(interval)
    elif len(sys.argv) > 1 and sys.argv[1] == "test":
        # Test: build briefs for all today's games and print (no delivery)
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
        games = _get_todays_games()
        goalie_matchups = get_starting_goalies(current_sports_date())
        try:
            injury_data = nba_injuries()
        except Exception:
            injury_data = []
        for g in games:
            if g["sport"] == "NHL":
                brief = build_nhl_brief(g, goalie_matchups)
            else:
                brief = build_nba_brief(g, injury_data)
            if brief:
                print(_format_brief_message(brief))
                print("---")
    else:
        # Single cycle
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
        hydrate_state()
        briefs = run_cycle()
        print(f"Delivered {len(briefs)} brief(s)")
        for b in briefs:
            print(f"  {b.away_team} @ {b.home_team} ({b.sport})")
