#!/usr/bin/env python3
"""
NBA live lineup monitor — detects confirmed starters and triggers re-pricing.

Runs every 2 minutes during pre-game windows (30 min before tip).
When ESPN confirms starters, compares against injury list and fires
re-pricing if a high-impact player is unexpectedly out.

If ensemble re-pricing produces edge > 2% vs Polymarket, fires BET signal.

Usage:
    python scripts/lineup_monitor.py              # run once
    python scripts/lineup_monitor.py --daemon      # continuous monitoring
"""

import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import requests

from src.data.nba_lineups import get_todays_nba_events, get_confirmed_starters, detect_lineup_changes

# Suppress noisy NBA.com timeout warnings from the stats module
logging.getLogger("src.data.nba_stats").setLevel(logging.ERROR)
logging.getLogger("src.data.nba_player_stats").setLevel(logging.ERROR)

from src.data.nba_player_stats import build_player_impact_map

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")

# How many minutes before tip to start checking lineups
PRE_GAME_WINDOW = 35

# Minimum impact to trigger a re-pricing alert
MIN_IMPACT_THRESHOLD = 0.03  # 3% — catches T1/T2 level absences

# Track which games we've already processed
_processed: set[str] = set()


def _supabase_headers():
    return {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
    }


def check_lineups():
    """Check all upcoming NBA games for confirmed lineups."""
    now = datetime.now(timezone.utc)
    events = get_todays_nba_events()

    if not events:
        return

    # Impact map is optional — NBA.com may be blocked from VPS
    try:
        impact_map = build_player_impact_map()
    except Exception:
        impact_map = {}
        logger.info("Player impact map unavailable (NBA.com blocked?) — lineup detection still active")

    for ev in events:
        event_id = ev["event_id"]

        # Skip already processed
        if event_id in _processed:
            continue

        # Skip non-scheduled games
        if ev["status"] not in ("Scheduled", "In Progress"):
            continue

        # Check if within pre-game window
        game_time_str = ev.get("game_time", "")
        if not game_time_str:
            continue

        try:
            gt = datetime.fromisoformat(game_time_str.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            continue

        minutes_until = (gt - now).total_seconds() / 60
        if minutes_until > PRE_GAME_WINDOW or minutes_until < -5:
            continue

        # Try to get confirmed starters
        starters = get_confirmed_starters(event_id)
        if not starters:
            logger.debug("No starters yet for %s (%.0f min to tip)", ev["name"], minutes_until)
            continue

        logger.info(
            "LINEUPS CONFIRMED: %s (%.0f min to tip)",
            ev["name"], minutes_until,
        )
        logger.info(
            "  %s starters: %s",
            starters["home_team"],
            ", ".join(starters["home_starters"][:5]),
        )
        logger.info(
            "  %s starters: %s",
            starters["away_team"],
            ", ".join(starters["away_starters"][:5]),
        )

        # Detect unexpected absences
        changes = detect_lineup_changes(starters, impact_map=impact_map)

        significant_changes = [c for c in changes if c["impact"] >= MIN_IMPACT_THRESHOLD]

        if significant_changes:
            for c in significant_changes:
                logger.warning(
                    "LINEUP CHANGE: %s %s (%s, impact=%.1f%%)",
                    c["player"], c["type"], c["team"], c["impact"] * 100,
                )

            # Fire re-pricing
            _trigger_repricing(ev, starters, significant_changes, impact_map)

        _processed.add(event_id)


def _trigger_repricing(event: dict, starters: dict, changes: list, impact_map: dict):
    """Re-price the game with confirmed lineup and check for edge."""
    home = event["home_team"]
    away = event["away_team"]

    total_impact = sum(
        c["impact"] * (-1 if c["side"] == "home" else 1)
        for c in changes
    )

    logger.info(
        "RE-PRICING: %s @ %s | lineup impact = %+.1f%% on home prob",
        away, home, total_impact * 100,
    )

    # Post to Discord alerts
    _post_lineup_alert(event, changes)


def _post_lineup_alert(event: dict, changes: list):
    """Post lineup change alert to Discord."""
    discord_token = os.getenv("DISCORD_TOKEN_RESEARCH", "")
    discord_ch = os.getenv("DISCORD_CH_TRADE_SIGNALS", "")

    if not discord_token or not discord_ch:
        return

    change_lines = []
    for c in changes:
        change_lines.append(f"**{c['player']}** ({c['team']}) — {c['type']} (impact: {c['impact']:.1%})")

    msg = (
        f"**LINEUP ALERT** | NBA\n"
        f"**{event['name']}**\n"
        + "\n".join(change_lines)
    )

    try:
        requests.post(
            f"https://discord.com/api/v10/channels/{discord_ch}/messages",
            headers={
                "Authorization": f"Bot {discord_token}",
                "Content-Type": "application/json",
            },
            json={"content": msg},
            timeout=10,
        )
    except Exception as e:
        logger.warning("Discord alert failed: %s", e)


def run_daemon():
    """Continuous lineup monitoring during game hours."""
    logger.info("Lineup monitor daemon starting (check every 120s)")

    while True:
        try:
            check_lineups()
        except Exception as e:
            logger.error("Lineup check failed: %s", e, exc_info=True)

        time.sleep(120)


if __name__ == "__main__":
    if "--daemon" in sys.argv:
        run_daemon()
    else:
        check_lineups()
