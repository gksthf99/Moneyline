"""
Closing Line Value (CLV) tracking system.

Records model probability at bet decision time, captures Polymarket
closing price at game tip-off, and calculates CLV in basis points.

Positive CLV = model beat the market's final assessment (genuine edge).
Negative CLV = market moved against the bet (overpaid).

Circuit breaker: if rolling 7-day CLV < -50 bps, stop betting.
"""

import logging
import os
from datetime import datetime, timedelta, timezone

import requests

logger = logging.getLogger(__name__)

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")

CLV_STOP_THRESHOLD_BPS = -50
CLV_WINDOW_DAYS = 7
CLV_MIN_SAMPLE = 10  # Need at least 10 games before circuit breaker activates


def _supabase_headers() -> dict:
    return {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
    }


def record_decision(
    game_id: int,
    sport: str,
    home_team: str,
    away_team: str,
    team_picked: str,
    model_prob: float,
    market_price: float,
    trade_id: int | None = None,
) -> bool:
    """Record model probability and market price at decision time.

    Called from research_agent when recommendation is BET,
    and from executor after successful trade placement.

    Returns True if recorded successfully.
    """
    if not SUPABASE_URL or not SUPABASE_KEY:
        logger.warning("Supabase not configured — skipping CLV record")
        return False

    row = {
        "game_id": game_id,
        "trade_id": trade_id,
        "sport": sport,
        "home_team": home_team,
        "away_team": away_team,
        "team_picked": team_picked,
        "model_prob": round(model_prob, 4),
        "market_price": round(market_price, 4),
        "decision_time": datetime.now(timezone.utc).isoformat(),
    }

    try:
        resp = requests.post(
            f"{SUPABASE_URL}/rest/v1/clv",
            headers={**_supabase_headers(), "Prefer": "return=minimal"},
            json=row,
            timeout=10,
        )
        if resp.status_code in (200, 201):
            logger.info("CLV decision recorded: game %d, %s @ %.1f%%", game_id, team_picked, market_price * 100)
            return True
        else:
            logger.warning("CLV record failed: %s %s", resp.status_code, resp.text)
            return False
    except Exception as e:
        logger.warning("CLV record error: %s", e)
        return False


def calculate_clv(market_price_at_decision: float, closing_price: float) -> dict:
    """Calculate CLV from decision price and closing price.

    CLV = (closing - decision) / decision * 10000 (in basis points)
    Positive = got a better price than the market's final view.
    """
    if market_price_at_decision <= 0:
        return {"clv_bps": 0.0, "clv_pct": 0.0}

    diff = closing_price - market_price_at_decision
    clv_pct = diff / market_price_at_decision
    clv_bps = clv_pct * 10000

    return {"clv_bps": round(clv_bps, 1), "clv_pct": round(clv_pct, 4)}


def capture_closing_price(game_id: int, closing_price: float) -> bool:
    """Update a CLV record with the closing price and compute CLV.

    Called by clv_capture.py cron script at game tip-off.
    """
    if not SUPABASE_URL or not SUPABASE_KEY:
        return False

    now = datetime.now(timezone.utc).isoformat()

    # Fetch existing record to get decision price
    try:
        resp = requests.get(
            f"{SUPABASE_URL}/rest/v1/clv",
            headers=_supabase_headers(),
            params={
                "game_id": f"eq.{game_id}",
                "closing_price": "is.null",
                "select": "id,market_price",
                "limit": "1",
            },
            timeout=10,
        )
        rows = resp.json()
        if not rows:
            return False

        row = rows[0]
        clv = calculate_clv(float(row["market_price"]), closing_price)

        # Update with closing price and CLV
        resp = requests.patch(
            f"{SUPABASE_URL}/rest/v1/clv",
            headers={**_supabase_headers(), "Prefer": "return=minimal"},
            params={"id": f"eq.{row['id']}"},
            json={
                "closing_price": round(closing_price, 4),
                "closing_time": now,
                "clv_bps": clv["clv_bps"],
                "clv_pct": clv["clv_pct"],
            },
            timeout=10,
        )
        if resp.status_code in (200, 204):
            logger.info("CLV closing captured: game %d, close=%.1f%%, CLV=%+.0f bps",
                        game_id, closing_price * 100, clv["clv_bps"])
            return True
        else:
            logger.warning("CLV close update failed: %s", resp.text)
            return False
    except Exception as e:
        logger.warning("CLV capture error: %s", e)
        return False


def backfill_results(game_id: int, result: str) -> bool:
    """Update CLV record with game result ('win' or 'loss').

    Called by performance_agent after grading.
    """
    if not SUPABASE_URL or not SUPABASE_KEY:
        return False

    try:
        resp = requests.patch(
            f"{SUPABASE_URL}/rest/v1/clv",
            headers={**_supabase_headers(), "Prefer": "return=minimal"},
            params={"game_id": f"eq.{game_id}"},
            json={"result": result},
            timeout=10,
        )
        return resp.status_code in (200, 204)
    except Exception as e:
        logger.warning("CLV result backfill error: %s", e)
        return False


def check_clv_health() -> dict:
    """Check rolling 7-day CLV health. Circuit breaker for betting.

    Returns:
        {
            "rolling_7d_clv_bps": float,
            "sample_size": int,
            "should_stop": bool,
            "reason": str,
        }
    """
    if not SUPABASE_URL or not SUPABASE_KEY:
        return {"rolling_7d_clv_bps": 0, "sample_size": 0, "should_stop": False, "reason": "No Supabase"}

    cutoff = (datetime.now(timezone.utc) - timedelta(days=CLV_WINDOW_DAYS)).isoformat()

    try:
        resp = requests.get(
            f"{SUPABASE_URL}/rest/v1/clv",
            headers=_supabase_headers(),
            params={
                "select": "clv_bps",
                "created_at": f"gte.{cutoff}",
                "clv_bps": "not.is.null",
            },
            timeout=10,
        )
        rows = resp.json()
        if not rows:
            return {"rolling_7d_clv_bps": 0, "sample_size": 0, "should_stop": False, "reason": "No CLV data yet"}

        clv_values = [float(r["clv_bps"]) for r in rows]
        avg_clv = sum(clv_values) / len(clv_values)
        n = len(clv_values)

        should_stop = avg_clv < CLV_STOP_THRESHOLD_BPS and n >= CLV_MIN_SAMPLE
        reason = ""
        if should_stop:
            reason = f"Rolling {CLV_WINDOW_DAYS}d CLV is {avg_clv:+.0f} bps on {n} games (threshold: {CLV_STOP_THRESHOLD_BPS} bps)"

        return {
            "rolling_7d_clv_bps": round(avg_clv, 1),
            "sample_size": n,
            "should_stop": should_stop,
            "reason": reason,
        }
    except Exception as e:
        logger.warning("CLV health check error: %s", e)
        return {"rolling_7d_clv_bps": 0, "sample_size": 0, "should_stop": False, "reason": f"Error: {e}"}
