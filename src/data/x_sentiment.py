"""
X (Twitter) sentiment feed via Nitter RSS.

Polls beat reporter accounts for injury reports, lineup changes, and
breaking news. Feeds Layer 3 (Information Edge) of the Research Agent.

Graceful degradation: if Nitter is unreachable, logs a warning and
returns an empty list — never crashes the Research Agent.
"""

import logging
import os
import re
from datetime import datetime, timezone, timedelta
from email.utils import parsedate_to_datetime

import requests
from bs4 import BeautifulSoup

from src.config import SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY

logger = logging.getLogger(__name__)

NITTER_BASE_URL = os.getenv("NITTER_BASE_URL", "https://nitter.net")

ACCOUNTS: dict[str, list[str]] = {
    "NBA": [
        "ShamsCharania",
        "wojespn",
        "ByTimReynolds",
        "IanBegley",
        "ChrisBHaynes",
    ],
    "NHL": [
        "PierreVLeBrun",
        "frank_seravalli",
        "DarrenDreger",
        "TSNBobMcKenzie",
    ],
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) "
        "Gecko/20100101 Firefox/128.0"
    ),
    "Accept": "application/rss+xml, application/xml, text/xml, */*",
}


def _parse_rss_timestamp(ts_str: str) -> datetime | None:
    """Parse RSS pubDate into timezone-aware datetime."""
    try:
        return parsedate_to_datetime(ts_str)
    except Exception:
        return None


def _strip_html(html: str) -> str:
    """Strip HTML tags, returning plain text."""
    soup = BeautifulSoup(html, "lxml")
    return soup.get_text(separator=" ", strip=True)


def _extract_url(item) -> str:
    """Extract post URL from RSS item."""
    link = item.find("link")
    guid = item.find("guid")
    if link and link.string:
        return link.string.strip()
    if guid and guid.string:
        return guid.string.strip()
    return ""


def fetch_account_posts(handle: str, max_posts: int = 10) -> list[dict]:
    """Fetch recent posts from a single account via Nitter RSS.

    Returns list of:
        {"handle": str, "text": str, "timestamp": str, "url": str}

    Returns empty list on any failure.
    """
    url = f"{NITTER_BASE_URL}/{handle}/rss"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        if resp.status_code != 200:
            logger.warning(
                "Nitter RSS returned %d for %s", resp.status_code, handle
            )
            return []
        if not resp.text.strip():
            logger.warning("Nitter RSS returned empty body for %s", handle)
            return []

        soup = BeautifulSoup(resp.text, "xml")
        items = soup.find_all("item")

        posts = []
        for item in items[:max_posts]:
            title = item.find("title")
            desc = item.find("description")
            pub_date = item.find("pubDate")

            raw_html = ""
            if desc and desc.string:
                raw_html = desc.string
            elif title and title.string:
                raw_html = title.string

            text = _strip_html(raw_html) if raw_html else ""
            if not text:
                continue

            ts = ""
            if pub_date and pub_date.string:
                parsed = _parse_rss_timestamp(pub_date.string.strip())
                ts = parsed.isoformat() if parsed else pub_date.string.strip()

            posts.append({
                "handle": handle,
                "text": text[:1000],
                "timestamp": ts,
                "url": _extract_url(item),
            })

        return posts

    except requests.RequestException as e:
        logger.warning("Nitter unreachable for %s: %s", handle, e)
        return []
    except Exception as e:
        logger.warning("Failed to parse RSS for %s: %s", handle, e)
        return []


def fetch_all_sentiment(sport: str, hours: float = 6.0) -> list[dict]:
    """Fetch posts from all accounts for a sport, filtered to recent window.

    Args:
        sport: "NBA" or "NHL"
        hours: only return posts from the last N hours (default 6)

    Returns deduplicated list sorted by timestamp descending.
    """
    sport = sport.upper()
    handles = ACCOUNTS.get(sport)
    if not handles:
        logger.warning("No accounts configured for sport: %s", sport)
        return []

    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    all_posts = []

    for handle in handles:
        posts = fetch_account_posts(handle)
        all_posts.extend(posts)

    # Filter to recent window
    filtered = []
    for post in all_posts:
        if not post["timestamp"]:
            continue
        try:
            ts = datetime.fromisoformat(post["timestamp"])
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            if ts >= cutoff:
                filtered.append(post)
        except (ValueError, TypeError):
            continue

    # Deduplicate by text content
    seen = set()
    unique = []
    for post in filtered:
        key = post["text"][:200]
        if key not in seen:
            seen.add(key)
            unique.append(post)

    # Sort by timestamp descending
    def sort_key(p):
        try:
            return datetime.fromisoformat(p["timestamp"])
        except (ValueError, TypeError):
            return datetime.min.replace(tzinfo=timezone.utc)

    unique.sort(key=sort_key, reverse=True)
    return unique


def store_sentiment_to_supabase(
    game_id: str,
    posts: list[dict],
) -> bool:
    """Store sentiment data to Supabase research table.

    Stores under source='x_nitter' with posts as JSONB metadata.
    Returns True on success, False on failure.
    """
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        logger.warning("Supabase credentials not configured, skipping store")
        return False

    if not posts:
        return True

    try:
        resp = requests.post(
            f"{SUPABASE_URL}/rest/v1/research",
            headers={
                "apikey": SUPABASE_SERVICE_ROLE_KEY,
                "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
                "Content-Type": "application/json",
                "Prefer": "return=minimal",
            },
            json={
                "game_id": game_id,
                "prob_decomposition": {
                    "source": "x_nitter",
                    "post_count": len(posts),
                    "posts": posts,
                    "fetched_at": datetime.now(timezone.utc).isoformat(),
                },
                "edge_type": "C",
                "effective_edge": 0.0,
                "recommendation": "MONITOR",
            },
            timeout=10,
        )
        resp.raise_for_status()
        return True
    except Exception as e:
        logger.warning("Failed to store sentiment to Supabase: %s", e)
        return False
