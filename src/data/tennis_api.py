"""
Tennis data client scaffolding.

Designed for ATP/WTA pre-match support:
- schedule and results
- player ratings / form
- match context
- market prices

The concrete upstream provider is configured by environment variables so the
repo can support different vendors without hardcoding a single dependency.
"""

from __future__ import annotations

import logging
from datetime import date

import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.config import (
    TENNIS_DATA_API_KEY,
    TENNIS_DATA_BASE_URL,
    TENNIS_ODDS_API_KEY,
    TENNIS_ODDS_BASE_URL,
    TENNIS_PROVIDER,
    TENNIS_SPORTRADAR_ACCESS_LEVEL,
    TENNIS_SPORTRADAR_LANGUAGE,
)
from src.data import cache

logger = logging.getLogger(__name__)


@retry(
    retry=retry_if_exception_type((requests.ConnectionError, requests.Timeout)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=3, min=3, max=20),
    before_sleep=lambda rs: logger.warning(
        "Tennis request failed (%s), retrying in %ds...",
        rs.outcome.exception(),
        rs.next_action.sleep,
    ),
)
def _get(url: str, params: dict | None = None, headers: dict | None = None) -> dict:
    response = requests.get(url, params=params, headers=headers, timeout=15)
    response.raise_for_status()
    return response.json()


def _data_auth_headers() -> dict[str, str]:
    headers: dict[str, str] = {}
    if TENNIS_DATA_API_KEY:
        headers["Authorization"] = f"Bearer {TENNIS_DATA_API_KEY}"
    return headers


def _odds_auth_headers() -> dict[str, str]:
    headers: dict[str, str] = {}
    if TENNIS_ODDS_API_KEY:
        headers["Authorization"] = f"Bearer {TENNIS_ODDS_API_KEY}"
    return headers


def _tennis_base_url() -> str:
    if TENNIS_DATA_BASE_URL:
        return TENNIS_DATA_BASE_URL.rstrip("/")
    if TENNIS_PROVIDER.lower() == "sportradar" and TENNIS_DATA_API_KEY:
        return (
            f"https://api.sportradar.com/tennis/"
            f"{TENNIS_SPORTRADAR_ACCESS_LEVEL}/v3/"
            f"{TENNIS_SPORTRADAR_LANGUAGE}"
        )
    return ""


def _inject_api_key(params: dict[str, str] | None = None) -> dict[str, str]:
    result = dict(params or {})
    if TENNIS_PROVIDER.lower() == "sportradar" and TENNIS_DATA_API_KEY:
        result["api_key"] = TENNIS_DATA_API_KEY
    return result


def _normalize_match(match: dict, *, default_tour: str | None = None) -> dict:
    player_a = (
        match.get("player_a")
        or match.get("home_team")
        or match.get("player1")
        or match.get("competitor_a")
        or ""
    )
    player_b = (
        match.get("player_b")
        or match.get("away_team")
        or match.get("player2")
        or match.get("competitor_b")
        or ""
    )
    return {
        "id": match.get("id") or match.get("match_id") or match.get("event_id"),
        "player_a_id": match.get("player_a_id"),
        "player_b_id": match.get("player_b_id"),
        "sport": "TENNIS",
        "tour": (match.get("tour") or default_tour or "ATP").upper(),
        "player_a": player_a,
        "player_b": player_b,
        "tournament": match.get("tournament") or match.get("event_name") or "",
        "round_name": match.get("round_name") or match.get("round") or "",
        "surface": (match.get("surface") or "hard").lower(),
        "best_of": int(match.get("best_of", 3) or 3),
        "scheduled_time": match.get("scheduled_time") or match.get("start_time") or match.get("date") or "",
        "indoor": bool(match.get("indoor", False)),
        "status": match.get("status") or "scheduled",
    }


def _normalize_sportradar_summary(summary: dict) -> dict:
    sport_event = summary.get("sport_event", {})
    competitors = sport_event.get("competitors", [])
    player_a = competitors[0] if len(competitors) > 0 else {}
    player_b = competitors[1] if len(competitors) > 1 else {}
    context = sport_event.get("sport_event_context", {})
    competition = context.get("competition", {})
    season = context.get("season", {})
    venue = sport_event.get("venue", {}) or {}
    conditions = summary.get("conditions", {}) or {}

    return _normalize_match(
        {
            "id": sport_event.get("id"),
            "player_a_id": player_a.get("id"),
            "player_b_id": player_b.get("id"),
            "player_a": player_a.get("name"),
            "player_b": player_b.get("name"),
            "tour": competition.get("gender") or season.get("name"),
            "tournament": competition.get("name"),
            "round_name": (sport_event.get("round") or {}).get("name") or context.get("round", {}).get("name"),
            "surface": conditions.get("ground") or venue.get("surface") or "hard",
            "best_of": sport_event.get("best_of") or 3,
            "scheduled_time": sport_event.get("start_time"),
            "indoor": bool(venue.get("indoor", False)),
            "status": sport_event.get("status"),
        }
    )


def _normalize_player_profile(profile: dict, *, player_name: str, tour: str) -> dict:
    return {
        "name": player_name,
        "tour": tour.upper(),
        "rating": float(profile.get("rating", 1500.0) or 1500.0),
        "surface_rating": float(profile.get("surface_rating", profile.get("rating", 1500.0)) or 1500.0),
        "recent_form": float(profile.get("recent_form", 0.5) or 0.5),
        "hold_pct": float(profile.get("hold_pct", 0.75) or 0.75),
        "break_pct": float(profile.get("break_pct", 0.22) or 0.22),
        "injury_risk": float(profile.get("injury_risk", 0.0) or 0.0),
    }


def _normalize_market(market: dict) -> dict:
    return {
        "player_a_price": market.get("player_a_price") or market.get("home_price"),
        "player_b_price": market.get("player_b_price") or market.get("away_price"),
        "player_a_ask": market.get("player_a_ask") or market.get("home_ask"),
        "player_b_ask": market.get("player_b_ask") or market.get("away_ask"),
        "player_a_bid": market.get("player_a_bid") or market.get("home_bid"),
        "player_b_bid": market.get("player_b_bid") or market.get("away_bid"),
        "market_id": market.get("market_id") or market.get("id"),
    }


def get_schedule(match_date: date | None = None, tour: str | None = None) -> list[dict]:
    base_url = _tennis_base_url()
    if not base_url:
        return []
    target_date = match_date or date.today()
    cache_key = f"{target_date.isoformat()}_{(tour or 'ALL').upper()}"
    cached = cache.get("schedule", "tennis_schedule", cache_key)
    if cached:
        return cached

    if TENNIS_PROVIDER.lower() == "sportradar":
        payload = _get(
            f"{base_url}/schedules/{target_date.isoformat()}/summaries.json",
            params=_inject_api_key(),
            headers=_data_auth_headers(),
        )
        matches = [_normalize_sportradar_summary(item) for item in payload.get("summaries", [])]
    else:
        payload = _get(
            f"{base_url}/schedule",
            params=_inject_api_key({"date": target_date.isoformat(), "tour": tour} if tour else {"date": target_date.isoformat()}),
            headers=_data_auth_headers(),
        )
        matches = [_normalize_match(item, default_tour=tour) for item in payload.get("matches", payload if isinstance(payload, list) else [])]
    if tour:
        matches = [match for match in matches if match.get("tour", "").upper().startswith(tour.upper())]
    cache.put("schedule", matches, "tennis_schedule", cache_key)
    return matches


def get_results(match_date: date | None = None, tour: str | None = None) -> list[dict]:
    base_url = _tennis_base_url()
    if not base_url:
        return []
    target_date = match_date or date.today()
    if TENNIS_PROVIDER.lower() == "sportradar":
        payload = _get(
            f"{base_url}/schedules/{target_date.isoformat()}/summaries.json",
            params=_inject_api_key(),
            headers=_data_auth_headers(),
        )
        matches = [_normalize_sportradar_summary(item) for item in payload.get("summaries", [])]
        results = [match for match in matches if (match.get("status") or "").lower() in {"closed", "ended", "complete", "finished"}]
    else:
        payload = _get(
            f"{base_url}/results",
            params=_inject_api_key({"date": target_date.isoformat(), "tour": tour} if tour else {"date": target_date.isoformat()}),
            headers=_data_auth_headers(),
        )
        results = [_normalize_match(item, default_tour=tour) for item in payload.get("matches", payload if isinstance(payload, list) else [])]
    if tour:
        results = [match for match in results if match.get("tour", "").upper().startswith(tour.upper())]
    return results


def get_player_profile(player_name: str, tour: str = "ATP", player_id: str | None = None) -> dict:
    base_url = _tennis_base_url()
    if not base_url:
        return _normalize_player_profile({}, player_name=player_name, tour=tour)
    cache_key = f"{tour.upper()}_{player_name.lower()}"
    cached = cache.get("player_stats", "tennis_player_profile", cache_key)
    if cached:
        return cached

    try:
        if TENNIS_PROVIDER.lower() == "sportradar" and player_id:
            payload = _get(
                f"{base_url}/competitors/{player_id}/profile.json",
                params=_inject_api_key(),
                headers=_data_auth_headers(),
            )
            profile = payload.get("competitor", payload)
            normalized = _normalize_player_profile(
                {
                    "rating": profile.get("ranking", 1500.0) or 1500.0,
                    "surface_rating": profile.get("surface_ranking", profile.get("ranking", 1500.0)),
                    "recent_form": profile.get("recent_form", 0.5),
                    "hold_pct": profile.get("hold_pct", 0.75),
                    "break_pct": profile.get("break_pct", 0.22),
                    "injury_risk": profile.get("injury_risk", 0.0),
                },
                player_name=profile.get("name", player_name),
                tour=tour,
            )
        else:
            payload = _get(
                f"{base_url}/players/profile",
                params=_inject_api_key({"name": player_name, "tour": tour.upper()}),
                headers=_data_auth_headers(),
            )
            normalized = _normalize_player_profile(payload.get("player", payload), player_name=player_name, tour=tour)
    except requests.RequestException:
        normalized = _normalize_player_profile({}, player_name=player_name, tour=tour)
    cache.put("player_stats", normalized, "tennis_player_profile", cache_key)
    return normalized


def get_match_market(match_id: str | None = None, *, player_a: str | None = None, player_b: str | None = None, tour: str | None = None) -> dict:
    if not TENNIS_ODDS_BASE_URL:
        return {}
    params: dict[str, str] = {}
    if match_id:
        params["match_id"] = match_id
    if player_a:
        params["player_a"] = player_a
    if player_b:
        params["player_b"] = player_b
    if tour:
        params["tour"] = tour.upper()
    payload = _get(
        f"{TENNIS_ODDS_BASE_URL.rstrip('/')}/match-market",
        params=_inject_api_key(params),
        headers=_odds_auth_headers(),
    )
    return _normalize_market(payload.get("market", payload))


def build_match_data(match: dict) -> dict:
    tour = (match.get("tour") or "ATP").upper()
    player_a = match.get("player_a", "")
    player_b = match.get("player_b", "")
    return {
        "player_a": get_player_profile(player_a, tour=tour, player_id=match.get("player_a_id")),
        "player_b": get_player_profile(player_b, tour=tour, player_id=match.get("player_b_id")),
        "context": {
            "player_a_rest_days": int(match.get("player_a_rest_days", 1) or 1),
            "player_b_rest_days": int(match.get("player_b_rest_days", 1) or 1),
            "player_a_last_match_minutes": int(match.get("player_a_last_match_minutes", 0) or 0),
            "player_b_last_match_minutes": int(match.get("player_b_last_match_minutes", 0) or 0),
            "player_a_travel_zones": int(match.get("player_a_travel_zones", 0) or 0),
            "player_b_travel_zones": int(match.get("player_b_travel_zones", 0) or 0),
        },
        "h2h": {
            "player_a_win_pct": match.get("h2h_player_a_win_pct"),
            "sample": int(match.get("h2h_sample", 0) or 0),
        },
        "market": get_match_market(
            match.get("id"),
            player_a=player_a,
            player_b=player_b,
            tour=tour,
        ),
    }
