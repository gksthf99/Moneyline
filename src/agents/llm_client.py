"""
OpenRouter LLM client with ZDR enforcement, prompt caching, and automatic fallback.

All LLM calls in the system go through this module.
"""

import os
import json
import logging
from dataclasses import dataclass

import requests

from src.config import SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY
from src.agents.model_router import ModelConfig

logger = logging.getLogger(__name__)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


@dataclass
class LLMResponse:
    content: str
    model_used: str
    usage: dict
    fallback_triggered: bool = False


class LLMClientError(Exception):
    """Raised when both primary and fallback models fail."""


def _build_headers() -> dict:
    return {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "X-OpenRouter-No-Store": "true",
    }


def _call_openrouter(
    model: str,
    messages: list[dict],
    effort: str = "medium",
    extended_thinking: bool = False,
    temperature: float = 0.3,
    max_tokens: int = 4096,
) -> dict:
    """Make a single OpenRouter API call. Raises on HTTP or parse errors."""
    payload: dict = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    if extended_thinking:
        payload["provider"] = {"require_parameters": True}
        payload["thinking"] = {"type": "enabled", "budget_tokens": max_tokens}

    resp = requests.post(
        f"{OPENROUTER_BASE_URL}/chat/completions",
        headers=_build_headers(),
        json=payload,
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


def _log_failure_to_supabase(
    agent: str,
    model: str,
    error: str,
) -> None:
    """Best-effort log of LLM failure to Supabase alerts table."""
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        return
    try:
        requests.post(
            f"{SUPABASE_URL}/rest/v1/alerts",
            headers={
                "apikey": SUPABASE_SERVICE_ROLE_KEY,
                "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
                "Content-Type": "application/json",
                "Prefer": "return=minimal",
            },
            json={
                "alert_type": "llm_failure",
                "source": agent,
                "content": json.dumps({"model": model, "error": error}),
                "delivery_channel": "internal",
                "delivered": True,
                "fallback_triggered": True,
            },
            timeout=10,
        )
    except Exception:
        logger.warning("Failed to log LLM failure to Supabase", exc_info=True)


def _extract_content(raw: dict) -> str:
    """Extract text content from OpenRouter response.

    Some models (e.g. kimi-k2.5) put output in reasoning fields
    instead of content. This handles both cases.
    """
    msg = raw["choices"][0]["message"]
    if msg.get("content"):
        return msg["content"]
    # Fallback: concatenate reasoning text (kimi-k2.5 pattern)
    reasoning = msg.get("reasoning") or ""
    if reasoning:
        return reasoning
    details = msg.get("reasoning_details") or []
    if details:
        return "".join(d.get("text", "") for d in details)
    return ""


def call(
    config: ModelConfig,
    messages: list[dict],
    agent_name: str = "unknown",
    temperature: float = 0.3,
    max_tokens: int = 4096,
) -> LLMResponse:
    """Call LLM with automatic fallback.

    Tries primary model first. On failure, retries with fallback_model
    if one is configured. Logs failures to Supabase alerts table.
    """
    # Primary attempt
    try:
        raw = _call_openrouter(
            model=config.model,
            messages=messages,
            effort=config.effort,
            extended_thinking=config.extended_thinking,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        choice = _extract_content(raw)
        return LLMResponse(
            content=choice,
            model_used=config.model,
            usage=raw.get("usage", {}),
        )
    except Exception as primary_err:
        logger.warning(
            "Primary model %s failed: %s", config.model, primary_err
        )
        _log_failure_to_supabase(agent_name, config.model, str(primary_err))

    # Fallback attempt
    if not config.fallback_model:
        raise LLMClientError(
            f"Primary model {config.model} failed and no fallback configured"
        )

    try:
        raw = _call_openrouter(
            model=config.fallback_model,
            messages=messages,
            effort=config.effort,
            extended_thinking=False,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        choice = _extract_content(raw)
        return LLMResponse(
            content=choice,
            model_used=config.fallback_model,
            usage=raw.get("usage", {}),
            fallback_triggered=True,
        )
    except Exception as fallback_err:
        logger.error(
            "Fallback model %s also failed: %s",
            config.fallback_model,
            fallback_err,
        )
        _log_failure_to_supabase(
            agent_name, config.fallback_model, str(fallback_err)
        )
        raise LLMClientError(
            f"Both {config.model} and {config.fallback_model} failed"
        ) from fallback_err
