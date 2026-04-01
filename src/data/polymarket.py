"""
Polymarket market data — Gamma API + CLOB API.

Gamma API (https://gamma-api.polymarket.com): market discovery, prices, metadata. No auth.
CLOB API (https://clob.polymarket.com): orderbook depth, real bid/ask. No auth for reads.

Series IDs:
  NBA: 10345
  NHL: 10346
"""

import json
import logging
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher

import requests

from src.data import cache

logger = logging.getLogger(__name__)

GAMMA_URL = "https://gamma-api.polymarket.com"
CLOB_URL = "https://clob.polymarket.com"

SERIES_IDS = {
    "NBA": "10345",
    "NHL": "10346",
}

# Team name aliases — Polymarket uses short names, our system uses full names
_TEAM_ALIASES = {
    # NBA
    "trail blazers": "portland trail blazers",
    "blazers": "portland trail blazers",
    "thunder": "oklahoma city thunder",
    "celtics": "boston celtics",
    "warriors": "golden state warriors",
    "nets": "brooklyn nets",
    "pacers": "indiana pacers",
    "raptors": "toronto raptors",
    "bulls": "chicago bulls",
    "jazz": "utah jazz",
    "timberwolves": "minnesota timberwolves",
    "wolves": "minnesota timberwolves",
    "clippers": "la clippers",
    "pelicans": "new orleans pelicans",
    "hawks": "atlanta hawks",
    "mavericks": "dallas mavericks",
    "mavs": "dallas mavericks",
    "lakers": "los angeles lakers",
    "rockets": "houston rockets",
    "nuggets": "denver nuggets",
    "grizzlies": "memphis grizzlies",
    "cavaliers": "cleveland cavaliers",
    "cavs": "cleveland cavaliers",
    "bucks": "milwaukee bucks",
    "76ers": "philadelphia 76ers",
    "sixers": "philadelphia 76ers",
    "knicks": "new york knicks",
    "heat": "miami heat",
    "magic": "orlando magic",
    "pistons": "detroit pistons",
    "hornets": "charlotte hornets",
    "wizards": "washington wizards",
    "spurs": "san antonio spurs",
    "suns": "phoenix suns",
    "kings": "sacramento kings",
    # NHL
    "devils": "new jersey devils",
    "rangers": "new york rangers",
    "penguins": "pittsburgh penguins",
    "hurricanes": "carolina hurricanes",
    "senators": "ottawa senators",
    "capitals": "washington capitals",
    "stars": "dallas stars",
    "avalanche": "colorado avalanche",
    "blues": "st. louis blues",
    "flames": "calgary flames",
    "flyers": "philadelphia flyers",
    "ducks": "anaheim ducks",
    "bruins": "boston bruins",
    "sabres": "buffalo sabres",
    "blackhawks": "chicago blackhawks",
    "blue jackets": "columbus blue jackets",
    "red wings": "detroit red wings",
    "panthers": "florida panthers",
    "canadiens": "montreal canadiens",
    "islanders": "new york islanders",
    "lightning": "tampa bay lightning",
    "maple leafs": "toronto maple leafs",
    "canucks": "vancouver canucks",
    "golden knights": "vegas golden knights",
    "jets": "winnipeg jets",
    "kraken": "seattle kraken",
    "sharks": "san jose sharks",
    "predators": "nashville predators",
    "wild": "minnesota wild",
    "oilers": "edmonton oilers",
    "kings": "los angeles kings",
    "utah": "utah hockey club",
}


def _normalize_team(name: str) -> str:
    """Normalize a team name for matching."""
    lower = name.lower().strip()
    return _TEAM_ALIASES.get(lower, lower)


def _match_team(poly_name: str, our_name: str) -> bool:
    """Check if a Polymarket team name matches our team name."""
    pn = _normalize_team(poly_name)
    on = our_name.lower().strip()
    # Exact match
    if pn == on:
        return True
    # One contains the other
    if pn in on or on in pn:
        return True
    # Fuzzy match on last word (mascot)
    poly_last = pn.split()[-1] if pn else ""
    our_last = on.split()[-1] if on else ""
    if poly_last and poly_last == our_last:
        return True
    return False


# ---------------------------------------------------------------------------
# Gamma API — market discovery + prices
# ---------------------------------------------------------------------------

def get_todays_markets(sport: str) -> list[dict]:
    """Get tonight's moneyline markets for a sport.

    Returns list of dicts with: home, away, home_price, away_price,
    liquidity, condition_id, token_ids, event_title, market_id.
    """
    series_id = SERIES_IDS.get(sport.upper())
    if not series_id:
        return []

    cache_key_suffix = f"polymarket_{sport}_{datetime.now(timezone.utc).strftime('%Y%m%d')}"
    cached = cache.get("market_odds", cache_key_suffix)
    if cached:
        return cached

    # Games today/tonight: end between now and +36h
    # 36h window catches late games that end past midnight UTC
    now = datetime.now(timezone.utc)
    end_min = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    end_max = (now + timedelta(hours=36)).strftime("%Y-%m-%dT%H:%M:%SZ")

    try:
        resp = requests.get(
            f"{GAMMA_URL}/events",
            params={
                "series_id": series_id,
                "active": "true",
                "closed": "false",
                "limit": "50",
                "end_date_min": end_min,
                "end_date_max": end_max,
            },
            timeout=15,
        )
        resp.raise_for_status()
        events = resp.json()
    except Exception as e:
        logger.warning("Failed to fetch Polymarket %s events: %s", sport, e)
        return []

    markets = []
    seen_events = set()

    for event in events:
        title = event.get("title", "")

        for m in event.get("markets", []):
            # Only moneyline markets (2 team outcomes)
            mt = m.get("marketType", "")
            if mt and mt != "moneyline":
                continue

            outcomes = json.loads(m.get("outcomes", "[]"))
            prices = json.loads(m.get("outcomePrices", "[]"))

            if len(outcomes) != 2 or len(prices) != 2:
                continue

            # Skip if both outcomes aren't team names (filter out over/under etc)
            # Heuristic: moneyline outcomes don't contain "Over"/"Under"/"Yes"/"No"
            if any(o in ("Over", "Under", "Yes", "No") for o in outcomes):
                continue

            # Dedup by event title (take highest liquidity moneyline)
            event_key = title
            if event_key in seen_events:
                continue
            seen_events.add(event_key)

            liq = float(m.get("liquidity", 0))

            # Token IDs: try clobTokenIds (array ordered same as outcomes)
            clob_ids = m.get("clobTokenIds")
            if isinstance(clob_ids, str):
                try:
                    clob_ids = json.loads(clob_ids)
                except (json.JSONDecodeError, TypeError):
                    clob_ids = []
            if not clob_ids:
                clob_ids = []

            token_ids = {}
            for i, outcome in enumerate(outcomes):
                if i < len(clob_ids):
                    token_ids[outcome] = clob_ids[i]

            markets.append({
                "event_title": title,
                "team_a": outcomes[0],
                "team_b": outcomes[1],
                "price_a": float(prices[0]),
                "price_b": float(prices[1]),
                "liquidity": liq,
                "condition_id": m.get("conditionId", ""),
                "market_id": m.get("id", ""),
                "token_ids": token_ids,
            })

    # Backfill token IDs from /markets endpoint if missing
    missing = [m for m in markets if not m["token_ids"]]
    if missing:
        market_ids = [m["market_id"] for m in missing if m["market_id"]]
        if market_ids:
            try:
                # Batch fetch market details
                for mid in market_ids:
                    resp = requests.get(
                        f"{GAMMA_URL}/markets",
                        params={"id": mid},
                        timeout=10,
                    )
                    if resp.status_code == 200:
                        details = resp.json()
                        if isinstance(details, list) and details:
                            detail = details[0]
                            clob_ids = detail.get("clobTokenIds")
                            if isinstance(clob_ids, str):
                                try:
                                    clob_ids = json.loads(clob_ids)
                                except (json.JSONDecodeError, TypeError):
                                    clob_ids = []
                            if clob_ids:
                                # Find matching market and set token IDs
                                for m in markets:
                                    if m["market_id"] == mid:
                                        outcomes = [m["team_a"], m["team_b"]]
                                        for i, o in enumerate(outcomes):
                                            if i < len(clob_ids):
                                                m["token_ids"][o] = clob_ids[i]
            except Exception as e:
                logger.warning("Failed to backfill token IDs: %s", e)

    cache.put("market_odds", markets, cache_key_suffix)
    return markets


def match_game_to_market(
    home_team: str,
    away_team: str,
    markets: list[dict],
) -> dict | None:
    """Match a game (home/away teams) to a Polymarket market.

    Returns dict with: home_price, away_price, liquidity, condition_id,
    token_ids, or None if no match.
    """
    for m in markets:
        team_a = m["team_a"]
        team_b = m["team_b"]

        # Try both orderings
        if _match_team(team_a, home_team) and _match_team(team_b, away_team):
            return {
                "home_price": m["price_a"],
                "away_price": m["price_b"],
                "home_outcome": team_a,
                "away_outcome": team_b,
                "liquidity": m["liquidity"],
                "condition_id": m["condition_id"],
                "market_id": m["market_id"],
                "token_ids": m["token_ids"],
            }
        elif _match_team(team_a, away_team) and _match_team(team_b, home_team):
            return {
                "home_price": m["price_b"],
                "away_price": m["price_a"],
                "home_outcome": team_b,
                "away_outcome": team_a,
                "liquidity": m["liquidity"],
                "condition_id": m["condition_id"],
                "market_id": m["market_id"],
                "token_ids": m["token_ids"],
            }

    return None


# ---------------------------------------------------------------------------
# CLOB API — real orderbook depth
# ---------------------------------------------------------------------------

def get_clob_price(token_id: str) -> dict | None:
    """Get CLOB best buy/sell prices + spread for a token.

    Returns: {buy_price, sell_price, spread, midpoint}
    or None on failure.
    """
    if not token_id:
        return None

    try:
        buy_resp = requests.get(
            f"{CLOB_URL}/price",
            params={"token_id": token_id, "side": "BUY"},
            timeout=10,
        )
        sell_resp = requests.get(
            f"{CLOB_URL}/price",
            params={"token_id": token_id, "side": "SELL"},
            timeout=10,
        )

        if buy_resp.status_code != 200 or sell_resp.status_code != 200:
            return None

        buy_price = float(buy_resp.json().get("price", 0))
        sell_price = float(sell_resp.json().get("price", 0))

        # Buy = best ask (what you pay to buy), Sell = best bid (what you get selling)
        return {
            "best_ask": buy_price,
            "best_bid": sell_price,
            "spread": buy_price - sell_price,
            "midpoint": (buy_price + sell_price) / 2,
        }
    except Exception as e:
        logger.warning("Failed to fetch CLOB price: %s", e)
        return None


def get_market_prices(home_team: str, away_team: str, sport: str) -> dict | None:
    """Full pipeline: find market, get prices, optionally get orderbook depth.

    Returns: {
        home_price, away_price, liquidity,
        home_bid, home_ask, away_bid, away_ask,  (from CLOB if available)
        spread,
    } or None if no market found.
    """
    markets = get_todays_markets(sport)
    match = match_game_to_market(home_team, away_team, markets)

    if not match:
        return None

    result = {
        "home_price": match["home_price"],
        "away_price": match["away_price"],
        "liquidity": match["liquidity"],
        "condition_id": match["condition_id"],
        "market_id": match["market_id"],
    }

    # Get CLOB bid/ask for real spread
    home_token = match["token_ids"].get(match["home_outcome"])
    away_token = match["token_ids"].get(match["away_outcome"])

    home_clob = get_clob_price(home_token) if home_token else None
    if home_clob:
        result["home_bid"] = home_clob["best_bid"]
        result["home_ask"] = home_clob["best_ask"]
        result["home_spread"] = home_clob["spread"]

    away_clob = get_clob_price(away_token) if away_token else None
    if away_clob:
        result["away_bid"] = away_clob["best_bid"]
        result["away_ask"] = away_clob["best_ask"]
        result["away_spread"] = away_clob["spread"]

    return result
