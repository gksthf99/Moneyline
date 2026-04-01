"""
Log a manual bet to Supabase.

Usage:
  python3 scripts/log_bet.py "Knicks" 0.94 1.00
  python3 scripts/log_bet.py "Knicks"              # defaults: market price auto-fetched, $1 bet
  python3 scripts/log_bet.py --list                 # show today's bets
  python3 scripts/log_bet.py --summary              # show all-time record
"""

import json
import os
import sys
from datetime import date, datetime, timedelta, timezone

import requests
from dotenv import load_dotenv
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")

HEADERS = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json",
}


def find_game(team_name: str) -> dict | None:
    """Find today's game matching the team name."""
    today = date.today()
    tomorrow = today + timedelta(days=1)
    lower = f"{today.isoformat()}T10:00:00Z"
    upper = f"{tomorrow.isoformat()}T10:00:00Z"

    resp = requests.get(
        f"{SUPABASE_URL}/rest/v1/games",
        headers=HEADERS,
        params={
            "and": f"(game_time.gte.{lower},game_time.lt.{upper})",
            "select": "id,sport,home_team,away_team,game_time",
        },
        timeout=10,
    )
    if resp.status_code != 200:
        return None

    needle = team_name.lower().strip()
    for g in resp.json():
        home = g.get("home_team", "").lower()
        away = g.get("away_team", "").lower()
        # Match by full name or last word (mascot)
        home_last = home.split()[-1] if home else ""
        away_last = away.split()[-1] if away else ""
        if needle in home or needle == home_last:
            return {**g, "team_picked": g["home_team"], "is_home": True}
        if needle in away or needle == away_last:
            return {**g, "team_picked": g["away_team"], "is_home": False}
    return None


def get_research(game_id: int) -> dict | None:
    """Get research record for a game (model probability)."""
    resp = requests.get(
        f"{SUPABASE_URL}/rest/v1/research",
        headers=HEADERS,
        params={
            "game_id": f"eq.{game_id}",
            "select": "prob_decomposition,recommendation,edge_type,effective_edge",
            "order": "id.desc",
            "limit": "1",
        },
        timeout=10,
    )
    if resp.status_code == 200 and resp.json():
        return resp.json()[0]
    return None


def log_bet(team_name: str, buy_price: float | None = None, amount: float = 1.0):
    """Log a manual bet."""
    game = find_game(team_name)
    if not game:
        print(f"No game found today for '{team_name}'")
        print("\nToday's games:")
        show_todays_games()
        return

    game_id = game["id"]
    sport = game["sport"]
    home = game["home_team"]
    away = game["away_team"]
    picked = game["team_picked"]

    # Get model data
    research = get_research(game_id)
    decomp = {}
    if research:
        d = research.get("prob_decomposition", {})
        if isinstance(d, str):
            d = json.loads(d)
        decomp = d

    final_prob = decomp.get("final", 0.5)
    model_prob = final_prob if game["is_home"] else 1.0 - final_prob
    base_prob = decomp.get("base", 0.0)
    sit_adj = decomp.get("situational", 0.0)
    info_edge = decomp.get("information", 0.0)

    # Auto-fetch market price if not provided
    if buy_price is None:
        try:
            from src.data.polymarket import get_market_prices
            mkt = get_market_prices(home, away, sport)
            if mkt:
                if game["is_home"]:
                    buy_price = mkt.get("home_ask") or mkt.get("home_price", 0)
                else:
                    buy_price = mkt.get("away_ask") or mkt.get("away_price", 0)
        except Exception:
            pass
        if not buy_price:
            buy_price = model_prob  # fallback to model prob

    edge_type = research.get("edge_type", "B") if research else "B"
    effective_edge = float(research.get("effective_edge", 0) or 0) if research else 0.0

    row = {
        "game_id": game_id,
        "sport": sport,
        "edge_type": edge_type,
        "base_prob": round(base_prob, 4),
        "situational_adj": round(sit_adj, 4),
        "info_edge": round(info_edge, 4),
        "final_prob": round(final_prob, 4),
        "true_implied": round(buy_price, 4),
        "effective_edge": round(effective_edge, 4),
        "position_size": round(amount, 2),
        "entry_price": round(buy_price, 4),
        "outcome": "pending",
        "team_picked": picked,
        "side": "BUY",
        "amount_usdc": round(amount, 2),
    }

    resp = requests.post(
        f"{SUPABASE_URL}/rest/v1/trades",
        headers={**HEADERS, "Prefer": "return=representation"},
        json=row,
        timeout=10,
    )
    if resp.status_code in (200, 201):
        trade = resp.json()[0] if resp.json() else {}
        print(f"Logged: {picked} ({sport}) @ {buy_price:.2f} | ${amount:.2f} | model: {model_prob:.0%}")
        print(f"  Game: {away} @ {home}")
        print(f"  Trade ID: {trade.get('id', '?')}")
    else:
        print(f"Failed to log: {resp.status_code} {resp.text}")


def show_todays_games():
    """List today's games."""
    today = date.today()
    tomorrow = today + timedelta(days=1)
    lower = f"{today.isoformat()}T10:00:00Z"
    upper = f"{tomorrow.isoformat()}T10:00:00Z"

    resp = requests.get(
        f"{SUPABASE_URL}/rest/v1/games",
        headers=HEADERS,
        params={
            "and": f"(game_time.gte.{lower},game_time.lt.{upper})",
            "select": "sport,home_team,away_team,game_time",
            "order": "game_time",
        },
        timeout=10,
    )
    if resp.status_code == 200:
        for g in resp.json():
            home_last = g["home_team"].split()[-1]
            away_last = g["away_team"].split()[-1]
            print(f"  [{g['sport']}] {away_last} @ {home_last}")


def show_list():
    """Show today's logged bets."""
    today = date.today()
    resp = requests.get(
        f"{SUPABASE_URL}/rest/v1/trades",
        headers=HEADERS,
        params={
            "created_at": f"gte.{today.isoformat()}",
            "select": "id,team_picked,entry_price,amount_usdc,outcome,sport,final_prob",
            "order": "created_at",
        },
        timeout=10,
    )
    if resp.status_code == 200:
        trades = resp.json()
        if not trades:
            print("No bets logged today")
            return
        print(f"Today's bets ({len(trades)}):")
        for t in trades:
            status = {"pending": "⏳", "win": "✅", "loss": "❌"}.get(t.get("outcome", ""), "?")
            print(f"  {status} {t['team_picked']} ({t.get('sport','')}) @ {t.get('entry_price',0):.2f} | ${t.get('amount_usdc',0):.2f} | model: {t.get('final_prob',0):.0%}")


def show_summary():
    """Show all-time betting summary."""
    resp = requests.get(
        f"{SUPABASE_URL}/rest/v1/trades",
        headers=HEADERS,
        params={
            "select": "outcome,amount_usdc,entry_price,pnl,team_picked,sport,created_at",
            "order": "created_at",
        },
        timeout=10,
    )
    if resp.status_code != 200:
        print(f"Failed: {resp.status_code}")
        return

    trades = resp.json()
    if not trades:
        print("No bets logged yet")
        return

    total = len(trades)
    wins = sum(1 for t in trades if t.get("outcome") == "win")
    losses = sum(1 for t in trades if t.get("outcome") == "loss")
    pending = sum(1 for t in trades if t.get("outcome") == "pending")
    total_pnl = sum(t.get("pnl", 0) or 0 for t in trades if t.get("pnl") is not None)
    total_wagered = sum(t.get("amount_usdc", 0) or 0 for t in trades)

    print(f"All-time record: {wins}W-{losses}L ({pending} pending)")
    print(f"Win rate: {wins/(wins+losses):.1%}" if (wins+losses) > 0 else "Win rate: N/A")
    print(f"Total wagered: ${total_wagered:.2f}")
    print(f"P&L: ${total_pnl:.2f}")
    if total_wagered > 0:
        print(f"ROI: {total_pnl/total_wagered:.1%}")

    # By sport
    for sport in ["NBA", "NHL"]:
        st = [t for t in trades if t.get("sport") == sport]
        if not st:
            continue
        sw = sum(1 for t in st if t.get("outcome") == "win")
        sl = sum(1 for t in st if t.get("outcome") == "loss")
        sp = sum(t.get("pnl", 0) or 0 for t in st if t.get("pnl") is not None)
        print(f"\n{sport}: {sw}W-{sl}L | P&L: ${sp:.2f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)

    if sys.argv[1] == "--list":
        show_list()
    elif sys.argv[1] == "--summary":
        show_summary()
    elif sys.argv[1] == "--games":
        show_todays_games()
    else:
        team = sys.argv[1]
        price = float(sys.argv[2]) if len(sys.argv) > 2 else None
        amt = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
        log_bet(team, price, amt)
