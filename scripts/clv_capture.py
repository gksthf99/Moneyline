#!/usr/bin/env python3
"""
Capture Polymarket closing prices at game tip-off.

Runs every 2 minutes during game hours via cron. For each CLV record
missing a closing price, checks if the game starts within 5 minutes.
If so, fetches the current Polymarket price and records it as the close.

Usage:
    python3 scripts/clv_capture.py
"""

import logging
import os
import sys
from datetime import datetime, timedelta, timezone

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.polymarket import get_market_prices
from src.model.clv import capture_closing_price

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")


def _supabase_headers():
    return {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
    }


def run():
    if not SUPABASE_URL or not SUPABASE_KEY:
        logger.error("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set")
        sys.exit(1)

    now = datetime.now(timezone.utc)

    # Find CLV records missing closing prices
    resp = requests.get(
        f"{SUPABASE_URL}/rest/v1/clv",
        headers=_supabase_headers(),
        params={
            "select": "id,game_id,sport,home_team,away_team,team_picked",
            "closing_price": "is.null",
        },
        timeout=10,
    )
    if resp.status_code != 200:
        logger.error("Failed to fetch CLV records: %s", resp.text)
        return

    pending = resp.json()
    if not pending:
        logger.info("No pending CLV records")
        return

    # Get game times for these game_ids
    game_ids = list({r["game_id"] for r in pending})
    id_filter = ",".join(str(gid) for gid in game_ids)
    resp = requests.get(
        f"{SUPABASE_URL}/rest/v1/games",
        headers=_supabase_headers(),
        params={
            "select": "id,game_time",
            "id": f"in.({id_filter})",
        },
        timeout=10,
    )
    games = {g["id"]: g for g in resp.json()}

    captured = 0
    for record in pending:
        game = games.get(record["game_id"])
        if not game or not game.get("game_time"):
            continue

        try:
            gt = datetime.fromisoformat(game["game_time"].replace("Z", "+00:00"))
        except (ValueError, TypeError):
            continue

        # Capture closing price if game starts within 5 minutes
        minutes_to_game = (gt - now).total_seconds() / 60
        if -5 <= minutes_to_game <= 5:
            # Fetch live price from Polymarket
            market = get_market_prices(
                record["home_team"], record["away_team"], record["sport"]
            )
            if not market:
                logger.warning("No market data for game %d", record["game_id"])
                continue

            # Get price for the team we picked
            is_home = record["team_picked"].lower().strip() == record["home_team"].lower().strip()
            if is_home:
                close_price = market.get("home_ask") or market.get("home_price", 0)
            else:
                close_price = market.get("away_ask") or market.get("away_price", 0)

            if close_price > 0:
                if capture_closing_price(record["game_id"], close_price):
                    captured += 1
                    logger.info(
                        "Captured close: game %d %s @ %.1f%%",
                        record["game_id"], record["team_picked"], close_price * 100,
                    )

    logger.info("Captured %d/%d closing prices", captured, len(pending))


if __name__ == "__main__":
    run()
