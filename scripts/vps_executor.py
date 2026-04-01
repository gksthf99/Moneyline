#!/usr/bin/env python3
"""
VPS-side trade executor.

Pulls today's BET signals from Supabase research table,
fetches live Polymarket prices, and executes trades.

Runs on the VPS (non-US IP) where Polymarket CLOB API is accessible.
Local machine handles all analysis and writes signals to Supabase.

Usage:
    python3 scripts/vps_executor.py          # execute today's BET signals
    python3 scripts/vps_executor.py --dry-run # show what would trade, no execution
"""

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.polymarket import get_market_prices, get_todays_markets, match_game_to_market
from src.trading.executor import Portfolio, execute_bet_signals

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")


def fetch_bet_signals() -> list[dict]:
    """Pull today's BET recommendations from Supabase research table."""
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set")

    # Use 10:00 UTC boundary (same as research agent) to get today's games
    now = datetime.now(timezone.utc)
    today_boundary = now.replace(hour=10, minute=0, second=0, microsecond=0)
    if now.hour < 10:
        from datetime import timedelta
        today_boundary -= timedelta(days=1)

    # Query research table for today's BET recommendations
    params = {
        "select": "id,game_id,recommendation,edge_type,effective_edge,prob_decomposition,created_at",
        "recommendation": "eq.BET",
        "created_at": f"gte.{today_boundary.isoformat()}",
        "order": "created_at.desc",
    }

    resp = requests.get(
        f"{SUPABASE_URL}/rest/v1/research",
        headers={
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
        },
        params=params,
        timeout=15,
    )
    resp.raise_for_status()
    signals = resp.json()

    if not signals:
        logger.info("No BET signals found for today")
        return []

    # Deduplicate by game_id (keep latest)
    seen = {}
    for s in signals:
        gid = s.get("game_id")
        if gid not in seen:
            seen[gid] = s
    signals = list(seen.values())

    logger.info("Found %d BET signal(s) from Supabase", len(signals))

    # Fetch game details for each signal
    game_ids = [s["game_id"] for s in signals if s.get("game_id")]
    games = _fetch_games(game_ids)

    enriched = []
    for sig in signals:
        game = games.get(sig["game_id"])
        if not game:
            logger.warning("No game found for game_id=%s, skipping", sig["game_id"])
            continue

        decomp = sig.get("prob_decomposition") or {}
        if isinstance(decomp, str):
            decomp = json.loads(decomp)

        final_prob = decomp.get("final") or decomp.get("final_probability") or 0.5
        base_prob = decomp.get("base") or decomp.get("base_probability") or 0.0
        sit_total = decomp.get("situational") or 0.0
        if isinstance(sit_total, dict):
            sit_total = sit_total.get("total", 0.0)
        sit_total = float(sit_total)
        info_total = decomp.get("information") or 0.0
        if isinstance(info_total, dict):
            info_total = info_total.get("total", 0.0)
        info_total = float(info_total)

        home = game.get("home_team", "")
        away = game.get("away_team", "")
        sport = game.get("sport", "NBA")

        # Determine which team to buy
        # Use bet_side from two-sided edge calculation when available
        bet_side = decomp.get("bet_side")
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

        # Calculate hours to game — skip games already started
        hours_to_game = 6.0
        game_time = game.get("game_time", "")
        if game_time:
            try:
                gt = datetime.fromisoformat(game_time.replace("Z", "+00:00"))
                raw_hours = (gt - datetime.now(timezone.utc)).total_seconds() / 3600
                if raw_hours < 0:
                    logger.info("Skipping %s @ %s — game already started", away, home)
                    continue
                hours_to_game = max(0.25, raw_hours)
            except (ValueError, TypeError):
                pass

        enriched.append({
            "game_id": sig["game_id"],
            "sport": sport,
            "home_team": home,
            "away_team": away,
            "team_to_buy": team_to_buy,
            "model_prob": model_prob,
            "hours_to_game": hours_to_game,
            "edge_type": sig.get("edge_type", "B"),
            "effective_edge": sig.get("effective_edge", 0.0),
            "base_prob": base_prob,
            "situational_adj": sit_total,
            "info_edge": info_total,
        })

    return enriched


def _fetch_games(game_ids: list[int]) -> dict:
    """Fetch game records from Supabase by IDs."""
    if not game_ids:
        return {}

    id_filter = ",".join(str(gid) for gid in game_ids)
    resp = requests.get(
        f"{SUPABASE_URL}/rest/v1/games",
        headers={
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
        },
        params={
            "select": "id,sport,home_team,away_team,game_time",
            "id": f"in.({id_filter})",
        },
        timeout=15,
    )
    resp.raise_for_status()
    return {g["id"]: g for g in resp.json()}


def _check_already_traded(game_ids: list[int]) -> set[int]:
    """Check which game_ids already have trades logged today."""
    if not game_ids:
        return set()

    now = datetime.now(timezone.utc)
    today_boundary = now.replace(hour=10, minute=0, second=0, microsecond=0)
    if now.hour < 10:
        from datetime import timedelta
        today_boundary -= timedelta(days=1)

    id_filter = ",".join(str(gid) for gid in game_ids)
    resp = requests.get(
        f"{SUPABASE_URL}/rest/v1/trades",
        headers={
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
        },
        params={
            "select": "game_id",
            "game_id": f"in.({id_filter})",
            "created_at": f"gte.{today_boundary.isoformat()}",
            "outcome": "eq.pending",
        },
        timeout=15,
    )
    resp.raise_for_status()
    return {t["game_id"] for t in resp.json()}


def _post_trade_signal(result, signal: dict):
    """Post executed trade signal to Discord with fill data + model predictions."""
    config_path = Path(__file__).resolve().parent.parent / ".discord_config.json"
    if not config_path.exists():
        return

    try:
        config = json.loads(config_path.read_text())
    except Exception:
        return

    sport = signal.get("sport", "").upper()
    home = signal.get("home_team", "")
    away = signal.get("away_team", "")
    model_prob = signal.get("model_prob", 0)
    base = signal.get("base_prob", 0)
    sit = signal.get("situational_adj", 0)
    info = signal.get("info_edge", 0)
    edge_type = signal.get("edge_type", "B")

    embed = {
        "title": f"TRADE — {away} @ {home}",
        "description": f"**{result.team_picked}** bought @ {result.price:.1%}",
        "color": 0x57F287,
        "fields": [
            {
                "name": "Fill",
                "value": f"${result.amount_usdc:.2f} @ {result.price:.1%}\nEdge: {result.effective_edge:+.1%}",
                "inline": True,
            },
            {
                "name": "Model",
                "value": f"Prediction: {model_prob:.1%}\nBase: {base:.1%} | Sit: {sit:+.1%} | Info: {info:+.1%}",
                "inline": True,
            },
            {
                "name": "Details",
                "value": f"Sport: {sport} | Type: {edge_type}\nOrder: `{result.order_id[:16]}...`",
                "inline": False,
            },
        ],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    sport_key = f"trade_signals_{sport.lower()}"
    webhook_url = (
        config.get("webhooks", {}).get(sport_key)
        or config.get("webhooks", {}).get("trade_signals")
    )
    if webhook_url:
        try:
            resp = requests.post(
                webhook_url,
                json={"username": "Trade Executor", "embeds": [embed]},
                timeout=10,
            )
            if resp.status_code in (200, 204):
                logger.info("Trade signal posted: %s", result.team_picked)
            else:
                logger.warning("Trade signal webhook failed: %s", resp.status_code)
        except Exception as e:
            logger.warning("Trade signal post error: %s", e)


def run(dry_run: bool = False):
    """Main execution loop."""
    # Validate env
    if not os.getenv("POLY_PRIVATE_KEY"):
        logger.error("POLY_PRIVATE_KEY not set — cannot execute trades")
        sys.exit(1)

    # CLV health check — log only (hard stop disabled, sample too small)
    try:
        from src.model.clv import check_clv_health
        clv = check_clv_health()
        if clv["sample_size"] > 0:
            logger.info("CLV health: %+.0f bps (n=%d)", clv["rolling_7d_clv_bps"], clv["sample_size"])
        if clv["should_stop"]:
            logger.warning("CLV WARNING: %s — trading continues (log-only mode)", clv["reason"])
    except Exception as e:
        logger.warning("CLV check failed (non-blocking): %s", e)

    # Fetch signals
    signals = fetch_bet_signals()
    if not signals:
        return

    # Check for already-traded games
    game_ids = [s["game_id"] for s in signals]
    already_traded = _check_already_traded(game_ids)
    if already_traded:
        logger.info("Skipping %d already-traded game(s): %s", len(already_traded), already_traded)
        signals = [s for s in signals if s["game_id"] not in already_traded]

    if not signals:
        logger.info("All signals already traded")
        return

    # Only trade games 15-45 min before tip-off for best CLV
    MIN_MINUTES = 15
    MAX_MINUTES = 45
    min_h, max_h = MIN_MINUTES / 60, MAX_MINUTES / 60
    in_window = [s for s in signals if min_h <= s["hours_to_game"] <= max_h]
    too_early = [s for s in signals if s["hours_to_game"] > max_h]
    too_late = [s for s in signals if s["hours_to_game"] < min_h]
    if too_early:
        logger.info("Deferring %d game(s) >%dm out: %s",
                     len(too_early), MAX_MINUTES,
                     ", ".join(f'{s["team_to_buy"]} ({s["hours_to_game"]*60:.0f}m)' for s in too_early))
    if too_late:
        logger.info("Skipping %d game(s) <%dm out (too close/started): %s",
                     len(too_late), MIN_MINUTES,
                     ", ".join(f'{s["team_to_buy"]} ({s["hours_to_game"]*60:.0f}m)' for s in too_late))
    signals = in_window

    if not signals:
        logger.info("No games in %d-%dm trade window — will retry later", MIN_MINUTES, MAX_MINUTES)
        return

    # Fetch live market data for each signal
    ready_signals = []
    for sig in signals:
        market_data = get_market_prices(sig["home_team"], sig["away_team"], sig["sport"])
        if not market_data:
            logger.warning("No market data for %s @ %s — skipping", sig["away_team"], sig["home_team"])
            continue

        market_data["_home_team"] = sig["home_team"]
        market_data["_away_team"] = sig["away_team"]
        market_data["_sport"] = sig["sport"]
        sig["market_data"] = market_data
        ready_signals.append(sig)

    if not ready_signals:
        logger.info("No signals with available market data")
        return

    # Display signals
    logger.info("=" * 60)
    for sig in ready_signals:
        md = sig["market_data"]
        is_home = sig["team_to_buy"].lower().strip() == sig["home_team"].lower().strip()
        if is_home:
            ask = md.get("home_ask") or md.get("home_price", 0)
        else:
            ask = md.get("away_ask") or md.get("away_price", 0)
        logger.info(
            "%s | %s @ %s | Buy: %s | Model: %.1f%% | Ask: %.1f%% | Edge: %.1f%% | Type: %s | %.1fh to game",
            sig["sport"], sig["away_team"], sig["home_team"],
            sig["team_to_buy"], sig["model_prob"] * 100,
            ask * 100, sig["effective_edge"] * 100,
            sig["edge_type"], sig["hours_to_game"],
        )
    logger.info("=" * 60)

    if dry_run:
        logger.info("DRY RUN — no trades executed")
        return

    # Execute — use live portfolio balance from Polymarket
    from src.trading.executor import get_live_bankroll
    bankroll = get_live_bankroll()
    logger.info("Live bankroll: $%.2f", bankroll)
    portfolio = Portfolio(starting_balance=bankroll)

    results = execute_bet_signals(ready_signals, portfolio)

    executed = sum(1 for r in results if r.success)
    failed = sum(1 for r in results if not r.success)
    total_spent = sum(r.amount_usdc for r in results if r.success)

    logger.info("Results: %d executed, %d failed, $%.2f deployed", executed, failed, total_spent)

    for i, r in enumerate(results):
        if r.success:
            logger.info("  OK: %s $%.2f @ %.3f (order %s)", r.team_picked, r.amount_usdc, r.price, r.order_id)
            # Post trade signal to Discord with actual fill + model predictions
            sig = ready_signals[i] if i < len(ready_signals) else {}
            _post_trade_signal(r, sig)
        else:
            logger.info("  SKIP: %s — %s", r.team_picked, r.reason)


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    run(dry_run=dry_run)
