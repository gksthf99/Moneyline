"""
Centralised model routing for all agents.

Research Agent: tiered by game triage level
  - A1 (deep): claude-opus-4-6, effort=high, extended thinking ON
  - Standard:  claude-sonnet-4-6, effort=medium, extended thinking ON
  - Skip:      no LLM call — return early

Performance Agent: kimi-k2.5, extended thinking OFF
  - Fallback:  claude-haiku-4-5-20251001

Alert Agent: kimi-k2.5, extended thinking OFF
  - Fallback:  claude-haiku-4-5-20251001
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    model: str
    provider: str
    effort: str
    extended_thinking: bool
    fallback_model: str | None = None


# ── Research Agent (tiered) ──────────────────────────────────────────

_RESEARCH_MODELS: dict[str, ModelConfig] = {
    "deep": ModelConfig(
        model="anthropic/claude-sonnet-4-6",
        provider="openrouter",
        effort="high",
        extended_thinking=False,
        fallback_model="moonshotai/kimi-k2.5",
    ),
    "standard": ModelConfig(
        model="anthropic/claude-sonnet-4-6",
        provider="openrouter",
        effort="medium",
        extended_thinking=False,
        fallback_model="moonshotai/kimi-k2.5",
    ),
}


def get_research_model(game_tier: str) -> ModelConfig | None:
    """Return model config for Research Agent based on game triage level.

    Returns None for skip-tier games (caller should return early).
    Raises ValueError for unknown tiers.
    """
    tier = game_tier.lower()
    if tier == "skip":
        return None
    if tier in _RESEARCH_MODELS:
        return _RESEARCH_MODELS[tier]
    raise ValueError(f"Unknown game tier: {game_tier!r}. Expected deep/standard/skip.")


# ── Performance Agent ────────────────────────────────────────────────

_PERFORMANCE_MODEL = ModelConfig(
    model="moonshotai/kimi-k2.5",
    provider="openrouter",
    effort="low",
    extended_thinking=False,
    fallback_model="anthropic/claude-haiku-4-5-20251001",
)


def get_performance_model() -> ModelConfig:
    return _PERFORMANCE_MODEL


# ── Alert Agent ──────────────────────────────────────────────────────

_ALERT_MODEL = ModelConfig(
    model="moonshotai/kimi-k2.5",
    provider="openrouter",
    effort="low",
    extended_thinking=False,
    fallback_model="anthropic/claude-haiku-4-5-20251001",
)


def get_alert_model() -> ModelConfig:
    return _ALERT_MODEL
