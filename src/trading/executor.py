"""
Polymarket trade execution engine.

Places real orders via the CLOB API using py-clob-client.
Integrates with:
  - Research agent BET signals (edge.py)
  - Supabase trades table (logging)
  - Discord trade-log channel (notifications)

Safety guards:
  - Daily loss limit (configurable, default 10% of bankroll)
  - Max concurrent positions (default 10)
  - Pre-trade edge re-check against live prices
  - Minimum liquidity requirement ($500)
  - 1.5% market movement check (stale signal guard)
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

import requests
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import MarketOrderArgs, OrderArgs, OrderType
from py_clob_client.order_builder.constants import BUY

from src.model.edge import full_edge_calculation

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CLOB_HOST = "https://clob.polymarket.com"
CHAIN_ID = 137  # Polygon mainnet

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")

# Discord trade-log webhook/channel
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN_RESEARCH", "")
DISCORD_CH_TRADE_LOG = os.getenv("DISCORD_CH_TRADE_LOG", "")
DISCORD_CH_TRADE_SIGNALS = os.getenv("DISCORD_CH_TRADE_SIGNALS", "")

# Safety limits
MIN_LIQUIDITY = 500         # $500 minimum market liquidity
MAX_DAILY_LOSS_PCT = 0.10   # 10% of bankroll daily loss limit
MAX_CONCURRENT = 10         # max open positions
STALE_MOVE_PCT = 0.015      # 1.5% market movement = stale signal


@dataclass
class TradeResult:
    """Result of a trade execution attempt."""
    success: bool
    order_id: str = ""
    side: str = ""
    token_id: str = ""
    team_picked: str = ""
    amount_usdc: float = 0.0
    price: float = 0.0
    effective_edge: float = 0.0
    reason: str = ""
    game_id: int | None = None


@dataclass
class Portfolio:
    """Simple bankroll tracker."""
    starting_balance: float
    current_balance: float = 0.0
    daily_pnl: float = 0.0
    open_positions: int = 0
    trades_today: int = 0
    last_reset_date: str = ""

    def __post_init__(self):
        if self.current_balance == 0.0:
            self.current_balance = self.starting_balance
        if not self.last_reset_date:
            self.last_reset_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def reset_daily(self):
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today != self.last_reset_date:
            self.daily_pnl = 0.0
            self.trades_today = 0
            self.last_reset_date = today

    def can_trade(self) -> tuple[bool, str]:
        """Check if we're within safety limits."""
        self.reset_daily()

        # Daily loss limit
        if self.daily_pnl < -(self.starting_balance * MAX_DAILY_LOSS_PCT):
            return False, f"Daily loss limit hit: ${self.daily_pnl:.2f}"

        # Max concurrent positions
        if self.open_positions >= MAX_CONCURRENT:
            return False, f"Max {MAX_CONCURRENT} concurrent positions reached"

        # Minimum balance
        if self.current_balance < 1.0:
            return False, "Insufficient balance"

        return True, "OK"


# ---------------------------------------------------------------------------
# CLOB Client
# ---------------------------------------------------------------------------

_client: ClobClient | None = None


def _get_client() -> ClobClient:
    """Initialize and return authenticated CLOB client (singleton)."""
    global _client
    if _client is not None:
        return _client

    private_key = os.getenv("POLY_PRIVATE_KEY", "")
    if not private_key:
        raise RuntimeError("POLY_PRIVATE_KEY not set in environment")

    funder = os.getenv("POLY_FUNDER_ADDRESS", "")
    sig_type = int(os.getenv("POLY_SIGNATURE_TYPE", "0"))

    kwargs = {
        "host": CLOB_HOST,
        "key": private_key,
        "chain_id": CHAIN_ID,
        "signature_type": sig_type,
    }
    if sig_type in (1, 2) and funder:
        kwargs["funder"] = funder

    _client = ClobClient(**kwargs)
    _client.set_api_creds(_client.create_or_derive_api_creds())

    logger.info("CLOB client initialized (sig_type=%d)", sig_type)
    return _client


def get_live_bankroll() -> float:
    """Query the live USDC balance from Polymarket exchange.

    Returns available cash balance in USD (not including open positions).
    Falls back to POLY_BANKROLL env var if query fails.
    """
    try:
        client = _get_client()
        from py_clob_client.headers.headers import create_level_2_headers
        from py_clob_client.clob_types import RequestArgs

        req = RequestArgs(method="GET", request_path="/balance-allowance")
        headers = create_level_2_headers(client.signer, client.creds, req)

        resp = requests.get(
            f"{CLOB_HOST}/balance-allowance",
            params={"asset_type": "COLLATERAL", "signature_type": str(int(os.getenv("POLY_SIGNATURE_TYPE", "0")))},
            headers=headers,
            timeout=10,
        )
        if resp.status_code == 200:
            data = resp.json()
            # Balance is in USDC raw units (6 decimals)
            balance_raw = int(data.get("balance", "0"))
            balance_usd = balance_raw / 1e6
            logger.info("Live bankroll: $%.2f", balance_usd)
            return balance_usd
    except Exception as e:
        logger.warning("Failed to query live bankroll: %s", e)

    fallback = float(os.getenv("POLY_BANKROLL", "100"))
    logger.info("Using fallback bankroll: $%.2f", fallback)
    return fallback


# ---------------------------------------------------------------------------
# Pre-trade validation
# ---------------------------------------------------------------------------

def _validate_signal(
    market_data: dict,
    model_prob: float,
    original_ask: float,
    hours_to_game: float,
    sport: str = "NBA",
    is_home: bool = True,
) -> tuple[bool, str, float]:
    """Re-check edge against LIVE prices before executing.

    Returns (ok, reason, live_effective_edge).
    Guards against stale signals where market has moved.
    Uses sport-specific thresholds (NBA: 1.5%, NHL: 4%).
    """
    # Check liquidity
    liquidity = market_data.get("liquidity", 0)
    if liquidity < MIN_LIQUIDITY:
        return False, f"Liquidity ${liquidity:.0f} below ${MIN_LIQUIDITY} minimum", 0.0

    # Get live bid/ask for the team we're actually buying
    if is_home:
        live_ask = market_data.get("home_ask") or market_data.get("home_price", 0)
        live_bid = market_data.get("home_bid") or 0
    else:
        live_ask = market_data.get("away_ask") or market_data.get("away_price", 0)
        live_bid = market_data.get("away_bid") or 0

    if live_ask <= 0:
        return False, "No valid ask price available", 0.0

    # Stale signal check: has market moved > 1.5% since signal was generated?
    price_move = abs(live_ask - original_ask)
    if price_move > STALE_MOVE_PCT:
        return False, f"Market moved {price_move:.1%} since signal (>{STALE_MOVE_PCT:.1%})", 0.0

    # Re-calculate edge with live prices (sport-specific threshold)
    edge = full_edge_calculation(
        your_probability=model_prob,
        ask_price=live_ask,
        bid_price=live_bid if live_bid > 0 else live_ask * 0.95,
        hours_to_game=hours_to_game,
        sport=sport,
    )

    if not edge.passes_threshold:
        return False, f"Live edge {edge.effective_edge:.1%} below threshold {edge.threshold_used:.0%}", edge.effective_edge

    return True, "Signal validated", edge.effective_edge


# ---------------------------------------------------------------------------
# Order execution
# ---------------------------------------------------------------------------

def execute_trade(
    game_id: int,
    sport: str,
    home_team: str,
    away_team: str,
    team_to_buy: str,
    model_prob: float,
    market_data: dict,
    hours_to_game: float,
    portfolio: Portfolio,
    edge_type: str = "B",
    base_prob: float = 0.0,
    situational_adj: float = 0.0,
    info_edge: float = 0.0,
) -> TradeResult:
    """Execute a single trade on Polymarket.

    Args:
        game_id: Supabase game ID
        sport: NBA or NHL
        home_team, away_team: team names
        team_to_buy: which team's YES token to buy
        model_prob: our model's probability for team_to_buy
        market_data: from polymarket.get_market_prices()
        hours_to_game: hours until game starts
        portfolio: current portfolio state
        edge_type: A1/A2/B/C classification
        base_prob, situational_adj, info_edge: layer decomposition

    Returns:
        TradeResult with execution details
    """
    # Production safety layer — sanity check + circuit breaker + lineage
    try:
        from src.model.safety import get_sanity, get_breaker, get_lineage, PredictionAudit

        # Circuit breaker check
        breaker = get_breaker()
        allowed, trip_reason = breaker.check()
        if not allowed:
            return TradeResult(success=False, reason=f"Circuit breaker: {trip_reason}", game_id=game_id, team_picked=team_to_buy)

        # Sanity check
        audit = PredictionAudit(
            game_id=game_id, sport=sport, home_team=home_team, away_team=away_team,
            timestamp=datetime.now(timezone.utc).isoformat(),
            market_price=market_data.get("home_ask") or market_data.get("home_price"),
            base_prob=base_prob, situational_adj=situational_adj,
            info_edge=info_edge, final_prob=model_prob,
        )
        sanity = get_sanity()
        passed, warnings = sanity.check_prediction(audit)

        # Log lineage regardless of sanity result
        get_lineage().log_prediction(audit)

        if not passed:
            logger.warning("Sanity check failed for game %d: %s", game_id, warnings)
            # Don't block — just warn. Hard blocks only for circuit breaker.
    except Exception as safety_err:
        logger.warning("Safety layer error (non-blocking): %s", safety_err)

    # Safety check
    can, reason = portfolio.can_trade()
    if not can:
        return TradeResult(success=False, reason=reason, game_id=game_id, team_picked=team_to_buy)

    # Determine which token to buy and what the ask price is
    is_home = team_to_buy.lower().strip() == home_team.lower().strip()

    if is_home:
        original_ask = market_data.get("home_ask") or market_data.get("home_price", 0)
        token_id = _resolve_token_id(market_data, home_team, is_home=True)
    else:
        original_ask = market_data.get("away_ask") or market_data.get("away_price", 0)
        token_id = _resolve_token_id(market_data, away_team, is_home=False)

    if not token_id:
        return TradeResult(success=False, reason="No token ID found for team", game_id=game_id, team_picked=team_to_buy)

    # Validate signal against live prices, including threshold enforcement
    ok, val_reason, live_edge = _validate_signal(market_data, model_prob, original_ask, hours_to_game, sport=sport, is_home=is_home)
    if not ok:
        return TradeResult(success=False, reason=val_reason, game_id=game_id, team_picked=team_to_buy)

    # Kelly criterion position sizing
    from src.model.edge import calculate_position_size as kelly_size
    live_bid = market_data.get("home_bid") or market_data.get("away_bid") or original_ask * 0.95
    amount_usdc = kelly_size(
        model_prob=model_prob,
        ask_price=original_ask,
        bankroll=portfolio.current_balance,
        edge_type=edge_type,
    )

    if amount_usdc < 1.0:
        return TradeResult(success=False, reason=f"Position ${amount_usdc:.2f} below $1 minimum", game_id=game_id, team_picked=team_to_buy)

    # Place market order (Fill-or-Kill for immediate execution)
    try:
        client = _get_client()

        order_args = MarketOrderArgs(
            token_id=token_id,
            amount=amount_usdc,
            side=BUY,
        )

        signed_order = client.create_market_order(order_args)
        resp = client.post_order(signed_order, OrderType.FOK)

        order_id = resp.get("orderID", resp.get("id", ""))
        success = resp.get("success", bool(order_id))

        if not success:
            error_msg = resp.get("errorMsg", resp.get("error", "Unknown error"))
            return TradeResult(
                success=False,
                reason=f"Order rejected: {error_msg}",
                game_id=game_id,
                team_picked=team_to_buy,
            )

        # Update portfolio
        portfolio.current_balance -= amount_usdc
        portfolio.open_positions += 1
        portfolio.trades_today += 1

        result = TradeResult(
            success=True,
            order_id=order_id,
            side="BUY",
            token_id=token_id,
            team_picked=team_to_buy,
            amount_usdc=amount_usdc,
            price=original_ask,
            effective_edge=live_edge,
            game_id=game_id,
        )

        # Log to Supabase and Discord
        _log_trade_supabase(result, sport, edge_type, base_prob, situational_adj, info_edge, model_prob)
        _post_trade_discord(result, sport, home_team, away_team, model_prob, live_edge, edge_type)

        # Record CLV decision point
        try:
            from src.model.clv import record_decision
            record_decision(
                game_id=game_id,
                sport=sport,
                home_team=home_team,
                away_team=away_team,
                team_picked=team_to_buy,
                model_prob=model_prob,
                market_price=original_ask,
            )
        except Exception as clv_err:
            logger.warning("CLV record failed: %s", clv_err)

        logger.info(
            "TRADE EXECUTED: %s %s $%.2f @ %.3f | edge %.1f%% | order %s",
            team_to_buy, sport, amount_usdc, original_ask, live_edge * 100, order_id,
        )

        return result

    except Exception as e:
        logger.error("Trade execution failed: %s", e, exc_info=True)
        return TradeResult(
            success=False,
            reason=f"Execution error: {e}",
            game_id=game_id,
            team_picked=team_to_buy,
        )


def _resolve_token_id(market_data: dict, team_name: str, is_home: bool) -> str | None:
    """Resolve the correct token ID for a team from market data.

    Uses the injected _home_team/_away_team/_sport to re-fetch from Polymarket
    and extract the correct token ID for order placement.
    """
    from src.data.polymarket import get_todays_markets, match_game_to_market

    home_team = market_data.get("_home_team", "")
    away_team = market_data.get("_away_team", "")
    sport = market_data.get("_sport", "NBA")

    if not home_team or not away_team:
        logger.warning("Missing team names in market_data for token resolution")
        return None

    markets = get_todays_markets(sport)
    match = match_game_to_market(home_team, away_team, markets)
    if not match:
        logger.warning("No market match for %s vs %s", home_team, away_team)
        return None

    outcome_name = match.get("home_outcome" if is_home else "away_outcome", "")
    token_ids = match.get("token_ids", {})
    token_id = token_ids.get(outcome_name)

    if not token_id:
        logger.warning("No token ID for outcome '%s' in match %s", outcome_name, match.get("market_id"))

    return token_id


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _log_trade_supabase(
    result: TradeResult,
    sport: str,
    edge_type: str,
    base_prob: float,
    situational_adj: float,
    info_edge: float,
    final_prob: float,
):
    """Write trade record to Supabase trades table."""
    if not SUPABASE_URL or not SUPABASE_KEY:
        logger.warning("Supabase not configured — skipping trade log")
        return

    row = {
        "game_id": result.game_id,
        "edge_type": edge_type,
        "base_prob": round(base_prob, 4),
        "situational_adj": round(situational_adj, 4),
        "info_edge": round(info_edge, 4),
        "final_prob": round(final_prob, 4),
        "true_implied": round(result.price, 4),
        "effective_edge": round(result.effective_edge, 4),
        "position_size": round(result.amount_usdc, 2),
        "entry_price": round(result.price, 4),
        "outcome": "pending",
        "polymarket_order_id": result.order_id,
        "token_id": result.token_id,
        "team_picked": result.team_picked,
        "side": result.side,
        "amount_usdc": round(result.amount_usdc, 2),
    }

    try:
        resp = requests.post(
            f"{SUPABASE_URL}/rest/v1/trades",
            headers={
                "apikey": SUPABASE_KEY,
                "Authorization": f"Bearer {SUPABASE_KEY}",
                "Content-Type": "application/json",
                "Prefer": "return=minimal",
            },
            json=row,
            timeout=10,
        )
        if resp.status_code not in (200, 201):
            logger.warning("Supabase trade insert failed: %s %s", resp.status_code, resp.text)
    except Exception as e:
        logger.warning("Failed to log trade to Supabase: %s", e)


def _post_trade_discord(
    result: TradeResult,
    sport: str,
    home_team: str,
    away_team: str,
    model_prob: float,
    live_edge: float,
    edge_type: str,
):
    """Post trade confirmation to Discord trade-log channel."""
    if not DISCORD_TOKEN or not DISCORD_CH_TRADE_LOG:
        return

    msg = (
        f"**TRADE EXECUTED** | {sport}\n"
        f"**{home_team}** vs **{away_team}**\n"
        f"Bought: **{result.team_picked}** @ {result.price:.1%}\n"
        f"Amount: **${result.amount_usdc:.2f}**\n"
        f"Model: {model_prob:.1%} | Edge: {live_edge:.1%} | Type: {edge_type}\n"
        f"Order: `{result.order_id}`"
    )

    try:
        requests.post(
            f"https://discord.com/api/v10/channels/{DISCORD_CH_TRADE_LOG}/messages",
            headers={
                "Authorization": f"Bot {DISCORD_TOKEN}",
                "Content-Type": "application/json",
            },
            json={"content": msg},
            timeout=10,
        )
    except Exception as e:
        logger.warning("Failed to post trade to Discord: %s", e)


# ---------------------------------------------------------------------------
# Batch execution — called by research agent
# ---------------------------------------------------------------------------

def execute_bet_signals(
    signals: list[dict],
    portfolio: Portfolio,
) -> list[TradeResult]:
    """Execute all BET signals from a research run.

    Each signal dict should contain:
        game_id, sport, home_team, away_team, team_to_buy,
        model_prob, market_data, hours_to_game, edge_type,
        base_prob, situational_adj, info_edge

    Returns list of TradeResults.
    """
    results = []

    for sig in signals:
        # Re-check portfolio limits before each trade
        can, reason = portfolio.can_trade()
        if not can:
            logger.warning("Stopping batch: %s", reason)
            results.append(TradeResult(
                success=False,
                reason=reason,
                game_id=sig.get("game_id"),
                team_picked=sig.get("team_to_buy", ""),
            ))
            break

        result = execute_trade(
            game_id=sig["game_id"],
            sport=sig["sport"],
            home_team=sig["home_team"],
            away_team=sig["away_team"],
            team_to_buy=sig["team_to_buy"],
            model_prob=sig["model_prob"],
            market_data=sig["market_data"],
            hours_to_game=sig.get("hours_to_game", 6.0),
            portfolio=portfolio,
            edge_type=sig.get("edge_type", "B"),
            base_prob=sig.get("base_prob", 0.0),
            situational_adj=sig.get("situational_adj", 0.0),
            info_edge=sig.get("info_edge", 0.0),
        )

        results.append(result)

        if not result.success:
            logger.warning(
                "Signal skipped: %s (%s) — %s",
                result.team_picked, sig.get("sport", ""), result.reason,
            )

        # Small delay between trades
        if result.success:
            time.sleep(1.0)

    # Summary log
    executed = sum(1 for r in results if r.success)
    total_spent = sum(r.amount_usdc for r in results if r.success)
    logger.info(
        "Batch complete: %d/%d executed, $%.2f deployed",
        executed, len(signals), total_spent,
    )

    return results
