from __future__ import annotations

import logging

import requests

from src.config import SUPABASE_SERVICE_ROLE_KEY, SUPABASE_URL

logger = logging.getLogger(__name__)


def is_configured() -> bool:
    return bool(SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY)


def headers(prefer: str | None = None) -> dict[str, str]:
    result = {
        "apikey": SUPABASE_SERVICE_ROLE_KEY or "",
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY or ''}",
    }
    if prefer:
        result["Prefer"] = prefer
    return result


def json_headers(prefer: str | None = None) -> dict[str, str]:
    result = headers(prefer=prefer)
    result["Content-Type"] = "application/json"
    return result


def select(table: str, params: dict, timeout: int = 10) -> list[dict]:
    if not is_configured():
        return []
    try:
        resp = requests.get(
            f"{SUPABASE_URL}/rest/v1/{table}",
            headers=headers(),
            params=params,
            timeout=timeout,
        )
        if resp.status_code == 200:
            return resp.json()
        logger.error("Supabase select failed [%s]: %s %s", table, resp.status_code, resp.text)
    except Exception as exc:
        logger.error("Supabase select error [%s]: %s", table, exc)
    return []


def insert(table: str, row: dict, timeout: int = 10) -> bool:
    if not is_configured():
        return False
    try:
        resp = requests.post(
            f"{SUPABASE_URL}/rest/v1/{table}",
            headers=json_headers(prefer="return=minimal"),
            json=row,
            timeout=timeout,
        )
        if resp.status_code in (200, 201):
            return True
        logger.error("Supabase insert failed [%s]: %s %s", table, resp.status_code, resp.text)
    except Exception as exc:
        logger.error("Supabase insert error [%s]: %s", table, exc)
    return False
