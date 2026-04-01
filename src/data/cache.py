"""
In-memory caching layer with TTL enforcement per data type.

Rules from gameplan:
- Team season stats: 24hr cache
- Injury reports: 30min cache with event invalidation
- Lineup / goalie starts: never cache — always live pull
- Historical H2H: 7 day cache
- Market odds: 5min cache
"""

import time
from typing import Any

from src.config import CACHE_TTL

_cache: dict[str, dict] = {}


def cache_key(data_type: str, *args) -> str:
    return f"{data_type}:{':'.join(str(a) for a in args)}"


def get(data_type: str, *args) -> Any | None:
    ttl = CACHE_TTL.get(data_type, 0)
    if ttl == 0:
        return None

    key = cache_key(data_type, *args)
    entry = _cache.get(key)
    if entry is None:
        return None

    if time.time() - entry["cached_at"] > ttl:
        del _cache[key]
        return None

    return entry["data"]


def put(data_type: str, data: Any, *args) -> None:
    ttl = CACHE_TTL.get(data_type, 0)
    if ttl == 0:
        return

    key = cache_key(data_type, *args)
    _cache[key] = {"data": data, "cached_at": time.time()}


def invalidate(data_type: str, *args) -> None:
    if args:
        key = cache_key(data_type, *args)
        _cache.pop(key, None)
    else:
        prefix = f"{data_type}:"
        to_delete = [k for k in _cache if k.startswith(prefix)]
        for k in to_delete:
            del _cache[k]


def clear_all() -> None:
    _cache.clear()
