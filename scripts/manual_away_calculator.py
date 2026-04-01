#!/usr/bin/env python3
"""
Manual away-side edge calculator — freeze period tool (March 27-31).

The current system only considers the favored side (home when prob >= 0.5,
away when prob < 0.5) for BET signals. For PASS/MONITOR games where the
model favors the away team (final_prob < 0.50), the away side may still
have exploitable edge that the executor never sees.

This script scans those overlooked away opportunities.

Modes:
  1. Auto-scan: pull today's research from Supabase, find away value
     python scripts/manual_away_calculator.py

  2. Manual: check a single game's away edge
     python scripts/manual_away_calculator.py --manual --home-prob 0.36 --away-ask 0.58 --sport NHL

  3. Dry-run (no Supabase, show what the scan logic does):
     python scripts/manual_away_calculator.py --dry-run
"""

import argparse
import json
import logging
import os
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

# Project root on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY
from src.model.edge import (
    full_edge_calculation,
    get_threshold,
    kelly_fraction,
    calculate_position_size,
    calculate_vig,
    calculate_true_implied,
    calculate_effective_edge,
    KELLY_FRACTION,
)
from src.data.polymarket import get_market_prices

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_FILE = LOG_DIR / "away_scanner.jsonl"

BANKROLL = 100.0  # default bankroll for Kelly sizing
SLIPPAGE = 0.005  # 0.5% slippage


# ---------------------------------------------------------------------------
# Supabase helpers
# ---------------------------------------------------------------------------

def _today_boundary() -> str:
    """10:00 UTC boundary for today's slate (same convention as research agent)."""
    now = datetime.now(timezone.utc)
    boundary = now.replace(hour=10, minute=0, second=0, microsecond=0)
    if now.hour < 10:
        boundary -= timedelta(days=1)
    return boundary.isoformat()


def fetch_research_signals() -> list[dict]:
    """Pull today's PASS and MONITOR research signals from Supabase."""
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        raise RuntimeError("SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY not set")

    boundary = _today_boundary()

    resp = requests.get(
        f"{SUPABASE_URL}/rest/v1/research",
        headers={
            "apikey": SUPABASE_SERVICE_ROLE_KEY,
            "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
        },
        params={
            "select": "id,game_id,recommendation,edge_type,effective_edge,prob_decomposition,created_at",
            "recommendation": "in.(PASS,MONITOR)",
            "created_at": f"gte.{boundary}",
            "order": "created_at.desc",
        },
        timeout=15,
    )
    resp.raise_for_status()
    signals = resp.json()

    # Deduplicate by game_id (keep latest)
    seen = {}
    for s in signals:
        gid = s.get("game_id")
        if gid and gid not in seen:
            seen[gid] = s
    return list(seen.values())


def fetch_games(game_ids: list[int]) -> dict[int, dict]:
    """Fetch game records by ID."""
    if not game_ids:
        return {}

    id_filter = ",".join(str(gid) for gid in game_ids)
    resp = requests.get(
        f"{SUPABASE_URL}/rest/v1/games",
        headers={
            "apikey": SUPABASE_SERVICE_ROLE_KEY,
            "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
        },
        params={
            "select": "id,sport,home_team,away_team,game_time",
            "id": f"in.({id_filter})",
        },
        timeout=15,
    )
    resp.raise_for_status()
    return {g["id"]: g for g in resp.json()}


# ---------------------------------------------------------------------------
# Core away-edge calculation
# ---------------------------------------------------------------------------

def calculate_away_edge(
    home_prob: float,
    away_ask: float,
    away_bid: float | None,
    sport: str,
    hours_to_game: float = 6.0,
    bankroll: float = BANKROLL,
) -> dict:
    """Calculate away-side edge for a single game.

    Args:
        home_prob: model's home-win probability (< 0.50 means model favors away)
        away_ask: market ask price for away token (what you'd pay)
        away_bid: market bid price for away token (None → estimate from ask)
        sport: NBA or NHL
        hours_to_game: hours until tip/puck drop
        bankroll: for Kelly sizing

    Returns dict with edge breakdown.
    """
    away_prob = 1.0 - home_prob

    # If no bid, estimate bid from ask (assume ~4 cent spread)
    if away_bid is None:
        away_bid = max(0.01, away_ask - 0.04)

    # Use full_edge_calculation from existing module
    edge_calc = full_edge_calculation(
        your_probability=away_prob,
        ask_price=away_ask,
        bid_price=away_bid,
        hours_to_game=hours_to_game,
        slippage_cost=SLIPPAGE,
        bankroll=bankroll,
        edge_type="B",
        sport=sport,
    )

    threshold = get_threshold(hours_to_game, sport=sport)

    return {
        "away_prob": away_prob,
        "away_ask": away_ask,
        "away_bid": away_bid,
        "vig": edge_calc.vig_estimate,
        "true_implied": edge_calc.true_implied,
        "effective_edge": edge_calc.effective_edge,
        "threshold": threshold,
        "passes_threshold": edge_calc.passes_threshold,
        "kelly_full": edge_calc.kelly_full,
        "kelly_used": edge_calc.kelly_used,
        "position_size": edge_calc.position_size,
        "reasoning": edge_calc.reasoning,
    }


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def format_hit(label: str, edge_data: dict, sport: str) -> str:
    """Format a HIT line."""
    e = edge_data
    lines = [
        f"  [HIT] {label} ({sport})",
        f"    Model: {1 - e['away_prob']:.1%} home / {e['away_prob']:.1%} away",
        f"    Away ask: {e['away_ask']:.2f} | True implied: {e['true_implied']:.1%} | Away edge: {e['effective_edge']:+.1%} (threshold: {e['threshold']:.0%})",
        f"    Kelly: ${e['position_size']:.2f} @ {e['away_ask']:.2f}",
    ]
    return "\n".join(lines)


def format_miss(label: str, edge_data: dict, sport: str) -> str:
    """Format a MISS line."""
    e = edge_data
    lines = [
        f"  [MISS] {label} ({sport})",
        f"    Model: {1 - e['away_prob']:.1%} home / {e['away_prob']:.1%} away",
        f"    Away ask: {e['away_ask']:.2f} | True implied: {e['true_implied']:.1%} | Away edge: {e['effective_edge']:+.1%} (threshold: {e['threshold']:.0%})",
        f"    Below threshold",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Log to JSONL
# ---------------------------------------------------------------------------

def log_opportunity(entry: dict) -> None:
    """Append an opportunity to the JSONL log."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ---------------------------------------------------------------------------
# Auto-scan mode
# ---------------------------------------------------------------------------

def run_auto_scan(bankroll: float = BANKROLL) -> None:
    """Pull today's PASS/MONITOR signals and scan for away-side value."""
    print("\n=== AWAY EDGE SCANNER ===\n")

    # 1. Fetch research signals
    try:
        signals = fetch_research_signals()
    except Exception as e:
        print(f"  ERROR: Failed to fetch research signals: {e}")
        return

    if not signals:
        print("  No PASS/MONITOR signals found for today.")
        return

    logger.info("Found %d PASS/MONITOR signal(s)", len(signals))

    # 2. Fetch game details
    game_ids = [s["game_id"] for s in signals if s.get("game_id")]
    games = fetch_games(game_ids)

    # 3. Filter to away-favoring games (final_prob < 0.50)
    candidates = []
    for sig in signals:
        game = games.get(sig.get("game_id"))
        if not game:
            continue

        decomp = sig.get("prob_decomposition") or {}
        if isinstance(decomp, str):
            try:
                decomp = json.loads(decomp)
            except (json.JSONDecodeError, TypeError):
                continue

        final_prob = decomp.get("final") or decomp.get("final_probability") or 0.5
        final_prob = float(final_prob)

        if final_prob >= 0.50:
            # Model favors home — not an away opportunity
            continue

        candidates.append({
            "game": game,
            "final_prob": final_prob,
            "recommendation": sig.get("recommendation", "MONITOR"),
        })

    if not candidates:
        print("  No away-favoring PASS/MONITOR games found today.")
        return

    print(f"  Scanning {len(candidates)} away-favoring game(s)...\n")

    hits = 0
    total = 0

    for c in candidates:
        game = c["game"]
        home = game["home_team"]
        away = game["away_team"]
        sport = game.get("sport", "NBA")
        label = f"{away} @ {home}"

        # Calculate hours to game
        hours_to_game = 6.0
        game_time = game.get("game_time", "")
        if game_time:
            try:
                gt = datetime.fromisoformat(game_time.replace("Z", "+00:00"))
                hours_to_game = max(0.5, (gt - datetime.now(timezone.utc)).total_seconds() / 3600)
            except (ValueError, TypeError):
                pass

        # Fetch market prices
        market = get_market_prices(home, away, sport)
        if not market:
            print(f"  [SKIP] {label} ({sport}) — no market found")
            continue

        away_ask = market.get("away_ask") or market.get("away_price")
        away_bid = market.get("away_bid")
        if not away_ask:
            print(f"  [SKIP] {label} ({sport}) — no away price")
            continue

        # Calculate edge
        edge_data = calculate_away_edge(
            home_prob=c["final_prob"],
            away_ask=away_ask,
            away_bid=away_bid,
            sport=sport,
            hours_to_game=hours_to_game,
            bankroll=bankroll,
        )

        total += 1

        if edge_data["passes_threshold"]:
            hits += 1
            print(format_hit(label, edge_data, sport))
        else:
            print(format_miss(label, edge_data, sport))

        print()

        # Log to JSONL
        log_entry = {
            "date": date.today().isoformat(),
            "game": label,
            "away_team": away,
            "home_team": home,
            "sport": sport,
            "model_home_prob": c["final_prob"],
            "model_away_prob": edge_data["away_prob"],
            "away_ask": edge_data["away_ask"],
            "true_implied": round(edge_data["true_implied"], 4),
            "away_edge": round(edge_data["effective_edge"], 4),
            "threshold": edge_data["threshold"],
            "above_threshold": edge_data["passes_threshold"],
            "kelly_position": edge_data["position_size"],
            "recommendation": c["recommendation"],
            "hours_to_game": round(hours_to_game, 1),
        }
        log_opportunity(log_entry)

    print(f"  Summary: {total} game(s) scanned, {hits} above threshold")
    if hits > 0:
        print(f"  Logged to: {LOG_FILE}")
    print()


# ---------------------------------------------------------------------------
# Manual mode
# ---------------------------------------------------------------------------

def run_manual(
    home_prob: float,
    away_ask: float,
    away_bid: float | None,
    sport: str,
    home_team: str = "Home",
    away_team: str = "Away",
    hours_to_game: float = 6.0,
    bankroll: float = BANKROLL,
) -> None:
    """Calculate and display away edge for a single manually-specified game."""
    print("\n=== AWAY EDGE CALCULATOR (MANUAL) ===\n")

    label = f"{away_team} @ {home_team}"

    edge_data = calculate_away_edge(
        home_prob=home_prob,
        away_ask=away_ask,
        away_bid=away_bid,
        sport=sport.upper(),
        hours_to_game=hours_to_game,
        bankroll=bankroll,
    )

    if edge_data["passes_threshold"]:
        print(format_hit(label, edge_data, sport.upper()))
    else:
        print(format_miss(label, edge_data, sport.upper()))

    print()

    # Log it
    log_entry = {
        "date": date.today().isoformat(),
        "game": label,
        "away_team": away_team,
        "home_team": home_team,
        "sport": sport.upper(),
        "model_home_prob": home_prob,
        "model_away_prob": edge_data["away_prob"],
        "away_ask": edge_data["away_ask"],
        "true_implied": round(edge_data["true_implied"], 4),
        "away_edge": round(edge_data["effective_edge"], 4),
        "threshold": edge_data["threshold"],
        "above_threshold": edge_data["passes_threshold"],
        "kelly_position": edge_data["position_size"],
        "recommendation": "MANUAL",
        "hours_to_game": hours_to_game,
    }
    log_opportunity(log_entry)
    print(f"  Logged to: {LOG_FILE}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Away-side edge scanner for PASS/MONITOR games",
    )
    parser.add_argument(
        "--manual", action="store_true",
        help="Manual single-game mode (requires --home-prob and --away-ask)",
    )
    parser.add_argument("--home-prob", type=float, help="Model's home-win probability (0-1)")
    parser.add_argument("--away-ask", type=float, help="Away token ask price (0-1)")
    parser.add_argument("--away-bid", type=float, default=None, help="Away token bid price (0-1)")
    parser.add_argument("--sport", type=str, default="NBA", help="NBA or NHL (default: NBA)")
    parser.add_argument("--home-team", type=str, default="Home", help="Home team name (for display)")
    parser.add_argument("--away-team", type=str, default="Away", help="Away team name (for display)")
    parser.add_argument("--hours", type=float, default=6.0, help="Hours to game (default: 6)")
    parser.add_argument("--bankroll", type=float, default=BANKROLL, help=f"Bankroll for Kelly sizing (default: {BANKROLL})")
    parser.add_argument("--dry-run", action="store_true", help="Show scan logic without Supabase (uses test data)")

    args = parser.parse_args()

    if args.manual:
        if args.home_prob is None or args.away_ask is None:
            parser.error("--manual requires --home-prob and --away-ask")

        run_manual(
            home_prob=args.home_prob,
            away_ask=args.away_ask,
            away_bid=args.away_bid,
            sport=args.sport,
            home_team=args.home_team,
            away_team=args.away_team,
            hours_to_game=args.hours,
            bankroll=args.bankroll,
        )
    elif args.dry_run:
        print("\n=== AWAY EDGE SCANNER (DRY RUN — test data) ===\n")

        test_cases = [
            {
                "label": "Wild @ Panthers",
                "sport": "NHL",
                "home_prob": 0.363,
                "away_ask": 0.58,
                "away_bid": 0.55,
            },
            {
                "label": "Avalanche @ Jets",
                "sport": "NHL",
                "home_prob": 0.340,
                "away_ask": 0.65,
                "away_bid": 0.62,
            },
            {
                "label": "Pacers @ Hawks",
                "sport": "NBA",
                "home_prob": 0.44,
                "away_ask": 0.53,
                "away_bid": 0.50,
            },
        ]

        hits = 0
        for tc in test_cases:
            edge_data = calculate_away_edge(
                home_prob=tc["home_prob"],
                away_ask=tc["away_ask"],
                away_bid=tc["away_bid"],
                sport=tc["sport"],
            )

            if edge_data["passes_threshold"]:
                hits += 1
                print(format_hit(tc["label"], edge_data, tc["sport"]))
            else:
                print(format_miss(tc["label"], edge_data, tc["sport"]))
            print()

        print(f"  Summary: {len(test_cases)} game(s) scanned, {hits} above threshold\n")
    else:
        run_auto_scan(bankroll=args.bankroll)


if __name__ == "__main__":
    main()
