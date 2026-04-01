"""
Production safety layer for the betting pipeline.

Four components:
1. SanityEngine — validates every prediction before bet submission
2. CircuitBreaker — stops betting if model degrades (Brier/CLV based)
3. LineageLogger — JSONL audit trail for every prediction
4. ShadowTesting — runs old vs new model in parallel for validation

Integration points:
- Before submit_bet(): SanityEngine.check_prediction()
- After daily close: CircuitBreaker.record_day()
- During model inference: LineageLogger.log_prediction()
- Next N games: ShadowTesting.predict_both()
"""

import json
import logging
import os
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger(__name__)

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")

LINEAGE_PATH = Path(__file__).resolve().parent.parent.parent / "logs" / "prediction_lineage.jsonl"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SANITY_CONFIG = {
    "max_daily_predictions": 25,     # NBA + NHL max games per day
    "min_prob": 0.05,                # matches our clamp
    "max_prob": 0.95,                # matches our clamp
    "max_favorite_rate": 0.45,       # max 45% of games >70% confidence
    "max_avg_confidence": 0.68,
    "min_avg_confidence": 0.32,
    "max_impact_adjustment": 0.15,   # matches our ±15% cap
    "max_market_divergence": 0.20,   # flag if model > 20% from market
    "max_net_rating_spread": 25.0,   # absurd team differential
}

BREAKER_CONFIG = {
    "max_daily_brier": 0.35,         # single-day disaster threshold
    "max_clv_divergence_bps": -500,  # 5-game CLV avg in bps
    "consecutive_bad_days": 2,       # trip after 2 bad days
}


# ---------------------------------------------------------------------------
# PredictionAudit
# ---------------------------------------------------------------------------

@dataclass
class PredictionAudit:
    """Complete audit record for a single game prediction."""
    game_id: int
    sport: str
    home_team: str
    away_team: str
    timestamp: str = ""

    # Raw inputs
    net_rating_home: float = 0.0
    net_rating_away: float = 0.0
    players_out_home: list = field(default_factory=list)
    players_out_away: list = field(default_factory=list)
    market_price: float | None = None

    # Calculated values
    base_prob: float = 0.0
    situational_adj: float = 0.0
    info_edge: float = 0.0
    final_prob: float = 0.0
    ensemble_prob: float = 0.0
    effective_edge: float = 0.0

    # Validation
    sanity_passed: bool = True
    warnings: list = field(default_factory=list)
    model_version: str = "v2_netrtg"

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


# ---------------------------------------------------------------------------
# SanityEngine
# ---------------------------------------------------------------------------

class SanityEngine:
    """Real-time validation of predictions before bet submission."""

    def __init__(self, config: dict | None = None):
        self.config = config or SANITY_CONFIG
        self._daily_predictions: list[PredictionAudit] = []
        self._daily_date: str = ""

    def check_prediction(self, audit: PredictionAudit) -> tuple[bool, list[str]]:
        """Validate single prediction. Returns (passed, warnings)."""
        warnings = []

        # 1. Probability bounds
        if not (self.config["min_prob"] <= audit.final_prob <= self.config["max_prob"]):
            warnings.append(
                f"PROB_OUT_OF_BOUNDS: {audit.final_prob:.3f} "
                f"not in [{self.config['min_prob']}, {self.config['max_prob']}]"
            )

        # 2. Impact adjustment magnitude
        if abs(audit.info_edge) > self.config["max_impact_adjustment"]:
            warnings.append(
                f"EXTREME_IMPACT: {audit.info_edge:.3f} exceeds ±{self.config['max_impact_adjustment']:.0%} cap"
            )

        # 3. Market price divergence
        if audit.market_price and audit.market_price > 0:
            divergence = abs(audit.final_prob - audit.market_price)
            if divergence > self.config["max_market_divergence"]:
                warnings.append(
                    f"MARKET_DIVERGENCE: {divergence:.3f} "
                    f"(model={audit.final_prob:.3f}, market={audit.market_price:.3f})"
                )

        # 4. Net rating spread sanity
        spread = audit.net_rating_home - audit.net_rating_away
        if abs(spread) > self.config["max_net_rating_spread"]:
            warnings.append(f"ABSURD_SPREAD: net rating diff {spread:.1f}")

        passed = len(warnings) == 0
        audit.sanity_passed = passed
        audit.warnings = warnings

        if not passed:
            for w in warnings:
                logger.warning("SANITY FAIL [game %d]: %s", audit.game_id, w)

        # Track for daily batch check
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today != self._daily_date:
            self._daily_predictions = []
            self._daily_date = today
        self._daily_predictions.append(audit)

        return passed, warnings

    def check_daily_batch(self) -> tuple[bool, dict]:
        """Validate today's accumulated predictions as a batch."""
        preds = self._daily_predictions
        if not preds:
            return True, {"count": 0}

        if len(preds) > self.config["max_daily_predictions"]:
            return False, {"error": "TOO_MANY_GAMES", "count": len(preds)}

        probs = [p.final_prob for p in preds]
        avg = sum(probs) / len(probs)
        high_conf = sum(1 for p in probs if p > 0.70 or p < 0.30)
        fav_rate = high_conf / len(probs)

        metrics = {
            "count": len(preds),
            "avg_prob": round(avg, 3),
            "favorite_rate": round(fav_rate, 3),
            "high_conf_count": high_conf,
            "sanity_pass_rate": round(sum(1 for p in preds if p.sanity_passed) / len(preds), 3),
        }

        failed = []
        if fav_rate > self.config["max_favorite_rate"]:
            failed.append("favorite_rate_too_high")
        if avg > self.config["max_avg_confidence"] or avg < self.config["min_avg_confidence"]:
            failed.append("avg_confidence_out_of_range")

        if failed:
            metrics["failed_checks"] = failed
            return False, metrics

        return True, metrics


# ---------------------------------------------------------------------------
# CircuitBreaker
# ---------------------------------------------------------------------------

class CircuitBreaker:
    """Stop betting if model is performing poorly. Integrates with existing CLV health check."""

    def __init__(self, config: dict | None = None):
        self.config = config or BREAKER_CONFIG
        self.daily_results: list[dict] = []
        self.is_open: bool = True  # True = trading allowed
        self.consecutive_bad_days: int = 0
        self.trip_reason: str = ""

    def record_day(self, date: datetime, brier: float, clv_avg: float):
        """Record end-of-day results and evaluate breaker state.

        Args:
            date: The date
            brier: Mean Brier score for the day
            clv_avg: Average CLV in percentage points (e.g., -5.0 = -5%)
        """
        self.daily_results.append({
            "date": date.isoformat() if hasattr(date, 'isoformat') else str(date),
            "brier": brier,
            "clv_avg": clv_avg,
        })
        self._evaluate()

    def _evaluate(self):
        if not self.daily_results:
            return

        latest = self.daily_results[-1]

        # Check 1: Single day disaster
        if latest["brier"] > self.config["max_daily_brier"]:
            self._trip(f"Disaster day: Brier {latest['brier']:.3f} > {self.config['max_daily_brier']}")
            return

        # Check 2: CLV divergence
        clv_bps = latest["clv_avg"] * 100  # convert % to bps
        if clv_bps < self.config["max_clv_divergence_bps"]:
            self.consecutive_bad_days += 1
        else:
            self.consecutive_bad_days = 0

        if self.consecutive_bad_days >= self.config["consecutive_bad_days"]:
            self._trip(f"CLV divergence for {self.consecutive_bad_days} consecutive days")

    def _trip(self, reason: str):
        self.is_open = False
        self.trip_reason = reason
        logger.error("CIRCUIT BREAKER TRIPPED: %s", reason)

        # Post to Discord
        _post_breaker_alert(reason)

    def reset(self):
        """Manual reset after investigation."""
        self.is_open = True
        self.consecutive_bad_days = 0
        self.trip_reason = ""
        logger.info("Circuit breaker manually reset")

    def check(self) -> tuple[bool, str]:
        """Check if trading is allowed. Returns (allowed, reason)."""
        return self.is_open, self.trip_reason


# ---------------------------------------------------------------------------
# LineageLogger
# ---------------------------------------------------------------------------

class LineageLogger:
    """JSONL audit trail for every prediction. Write-ahead to local file + Supabase."""

    def __init__(self):
        LINEAGE_PATH.parent.mkdir(parents=True, exist_ok=True)

    def log_prediction(self, audit: PredictionAudit):
        """Log complete prediction trail to JSONL and Supabase."""
        record = {
            **audit.to_dict(),
            "log_timestamp": datetime.now(timezone.utc).isoformat(),
            "schema_version": "2.0",
        }

        # Write to local JSONL (always succeeds)
        try:
            with open(LINEAGE_PATH, "a") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except Exception as e:
            logger.warning("JSONL write failed: %s", e)

        # Write to Supabase
        if SUPABASE_URL and SUPABASE_KEY:
            try:
                requests.post(
                    f"{SUPABASE_URL}/rest/v1/prediction_lineage",
                    headers={
                        "apikey": SUPABASE_KEY,
                        "Authorization": f"Bearer {SUPABASE_KEY}",
                        "Content-Type": "application/json",
                        "Prefer": "return=minimal",
                    },
                    json={
                        "game_id": audit.game_id,
                        "sport": audit.sport,
                        "home_team": audit.home_team,
                        "away_team": audit.away_team,
                        "base_prob": audit.base_prob,
                        "info_edge": audit.info_edge,
                        "final_prob": audit.final_prob,
                        "market_price": audit.market_price,
                        "effective_edge": audit.effective_edge,
                        "sanity_passed": audit.sanity_passed,
                        "warnings": json.dumps(audit.warnings),
                        "model_version": audit.model_version,
                        "raw_audit": json.dumps(record, default=str),
                    },
                    timeout=10,
                )
            except Exception as e:
                logger.warning("Supabase lineage write failed: %s", e)


# ---------------------------------------------------------------------------
# ShadowTesting
# ---------------------------------------------------------------------------

class ShadowTesting:
    """Run old model (win% Elo) alongside new model (net rating) for comparison."""

    def __init__(self, alert_threshold: float = 0.05):
        self.alert_threshold = alert_threshold
        self.comparisons: list[dict] = []

    def predict_both(
        self,
        game_id: int,
        home_team: str,
        away_team: str,
        old_prob: float,
        new_prob: float,
    ) -> dict:
        """Record both model outputs for a game.

        old_prob: win%-Elo model probability
        new_prob: net rating model probability
        """
        diff = abs(old_prob - new_prob)
        comparison = {
            "game_id": game_id,
            "home_team": home_team,
            "away_team": away_team,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "old_prob": round(old_prob, 4),
            "new_prob": round(new_prob, 4),
            "diff": round(diff, 4),
            "alert": diff > self.alert_threshold,
        }

        self.comparisons.append(comparison)

        if comparison["alert"]:
            logger.warning(
                "SHADOW ALERT: game %d %s @ %s | Old: %.3f | New: %.3f | Diff: %.3f",
                game_id, away_team, home_team, old_prob, new_prob, diff,
            )

        return comparison

    def validate(self, min_games: int = 10) -> tuple[bool, dict]:
        """Check if new model is safe for full cutover.

        Returns (safe, metrics).
        """
        if len(self.comparisons) < min_games:
            return False, {"reason": f"Only {len(self.comparisons)}/{min_games} games"}

        diffs = [c["diff"] for c in self.comparisons]
        alerts = [c for c in self.comparisons if c["alert"]]

        metrics = {
            "games": len(self.comparisons),
            "max_diff": round(max(diffs), 4),
            "mean_diff": round(sum(diffs) / len(diffs), 4),
            "alert_count": len(alerts),
            "alert_rate": round(len(alerts) / len(self.comparisons), 3),
        }

        # Safe if: no diff > 15% and alert rate < 20%
        safe = metrics["max_diff"] < 0.15 and metrics["alert_rate"] < 0.20

        logger.info("Shadow validation: %s — safe=%s", metrics, safe)
        return safe, metrics


# ---------------------------------------------------------------------------
# Singleton instances
# ---------------------------------------------------------------------------

_sanity: SanityEngine | None = None
_breaker: CircuitBreaker | None = None
_lineage: LineageLogger | None = None
_shadow: ShadowTesting | None = None


def get_sanity() -> SanityEngine:
    global _sanity
    if _sanity is None:
        _sanity = SanityEngine()
    return _sanity


def get_breaker() -> CircuitBreaker:
    global _breaker
    if _breaker is None:
        _breaker = CircuitBreaker()
    return _breaker


def get_lineage() -> LineageLogger:
    global _lineage
    if _lineage is None:
        _lineage = LineageLogger()
    return _lineage


def get_shadow() -> ShadowTesting:
    global _shadow
    if _shadow is None:
        _shadow = ShadowTesting()
    return _shadow


# ---------------------------------------------------------------------------
# Discord alert for circuit breaker
# ---------------------------------------------------------------------------

def _post_breaker_alert(reason: str):
    token = os.getenv("DISCORD_TOKEN_RESEARCH", "")
    channel = os.getenv("DISCORD_CH_TRADE_SIGNALS", "")
    if not token or not channel:
        return

    try:
        requests.post(
            f"https://discord.com/api/v10/channels/{channel}/messages",
            headers={"Authorization": f"Bot {token}", "Content-Type": "application/json"},
            json={"content": f"**CIRCUIT BREAKER TRIPPED**\n{reason}\nAll trading halted. Manual reset required."},
            timeout=10,
        )
    except Exception:
        pass
