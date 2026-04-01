"""
Trade settlement — grade pending trades after games complete.

Checks pending trades against game results, updates outcome and P&L.
Called by performance_agent after grading games.
"""

import logging
import os
from datetime import datetime, timezone

import requests

logger = logging.getLogger(__name__)

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")


def get_pending_trades() -> list[dict]:
    """Fetch all trades with outcome='pending' from Supabase."""
    if not SUPABASE_URL:
        return []

    try:
        resp = requests.get(
            f"{SUPABASE_URL}/rest/v1/trades",
            headers={
                "apikey": SUPABASE_KEY,
                "Authorization": f"Bearer {SUPABASE_KEY}",
            },
            params={
                "outcome": "eq.pending",
                "select": "*,games(home_team,away_team,sport,status)",
            },
            timeout=10,
        )
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        logger.warning("Failed to fetch pending trades: %s", e)
    return []


def settle_trade(trade: dict, home_won: bool) -> dict | None:
    """Settle a single trade based on game result.

    Returns updated trade dict or None on failure.
    """
    trade_id = trade.get("id")
    team_picked = trade.get("team_picked", "")
    game = trade.get("games", {})
    home_team = game.get("home_team", "")

    # Determine if our pick won
    picked_home = team_picked.lower().strip() == home_team.lower().strip()
    our_pick_won = (picked_home and home_won) or (not picked_home and not home_won)

    outcome = "win" if our_pick_won else "loss"

    # Calculate P&L
    entry_price = trade.get("entry_price", 0)
    amount_usdc = trade.get("amount_usdc", 0)

    if amount_usdc > 0 and entry_price > 0:
        shares = amount_usdc / entry_price
        if our_pick_won:
            # Each share pays $1 on win
            pnl = shares - amount_usdc  # profit = payout - cost
        else:
            pnl = -amount_usdc  # lost entire position
    else:
        pnl = 0.0

    # Brier contribution
    final_prob = trade.get("final_prob", 0.5)
    actual = 1.0 if our_pick_won else 0.0
    brier = (final_prob - actual) ** 2

    # Update in Supabase
    update = {
        "outcome": outcome,
        "pnl": round(pnl, 4),
        "brier_contribution": round(brier, 4),
        "settlement_price": 1.0 if our_pick_won else 0.0,
        "settled_at": datetime.now(timezone.utc).isoformat(),
    }

    try:
        resp = requests.patch(
            f"{SUPABASE_URL}/rest/v1/trades",
            headers={
                "apikey": SUPABASE_KEY,
                "Authorization": f"Bearer {SUPABASE_KEY}",
                "Content-Type": "application/json",
                "Prefer": "return=minimal",
            },
            params={"id": f"eq.{trade_id}"},
            json=update,
            timeout=10,
        )
        if resp.status_code not in (200, 204):
            logger.warning("Failed to settle trade %s: %s", trade_id, resp.text)
            return None
    except Exception as e:
        logger.warning("Failed to settle trade %s: %s", trade_id, e)
        return None

    logger.info(
        "Settled trade %s: %s %s | P&L: $%.2f | Brier: %.4f",
        trade_id, team_picked, outcome, pnl, brier,
    )

    return {**trade, **update}


def settle_all(game_results: dict[int, bool]) -> list[dict]:
    """Settle all pending trades given game results.

    Args:
        game_results: {game_id: home_won} mapping

    Returns:
        List of settled trade dicts.
    """
    pending = get_pending_trades()
    if not pending:
        logger.info("No pending trades to settle")
        return []

    settled = []
    for trade in pending:
        game_id = trade.get("game_id")
        if game_id not in game_results:
            continue

        result = settle_trade(trade, game_results[game_id])
        if result:
            settled.append(result)

    # Summary
    wins = sum(1 for s in settled if s.get("outcome") == "win")
    losses = sum(1 for s in settled if s.get("outcome") == "loss")
    total_pnl = sum(s.get("pnl", 0) for s in settled)
    logger.info(
        "Settlement complete: %d trades | %dW-%dL | P&L: $%.2f",
        len(settled), wins, losses, total_pnl,
    )

    return settled
