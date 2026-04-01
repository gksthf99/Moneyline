"""
R3: Online-learning ensemble model.

Combines all features (net rating, player impact, situational, market price, CLV)
into a calibrated win probability via logistic regression with online retraining.

Architecture:
  - Warm-starts with domain-knowledge weights (no backtest training needed)
  - Auto-retrains every 25 games with expanding window (max 150 games)
  - Feature 6 (market price) is the strongest predictor — model degrades without it
  - Falls back to heuristic blend when insufficient training data

Features:
  1. Net rating spread (home - away, from R1)
  2. Player impact adjustment sum (from R2)
  3. Rest differential (home_rest - away_rest, from L2)
  4. Home court (binary 1/0)
  5. Pace differential (home - away)
  6. Market price (Polymarket implied probability — most important)
  7. CLV residual (rolling 5-game model CLV, proxy for model accuracy trend)
"""

import json
import logging
import math
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")

# Retrain trigger: accumulate this many new games before retraining
RETRAIN_EVERY = 25

# Maximum training window (drop older games to prevent staleness)
MAX_TRAIN_WINDOW = 150

# Minimum games before switching from heuristic to learned weights
MIN_GAMES_TO_TRAIN = 45

# Model persistence path
MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "backtest_results" / "ensemble_model.json"


@dataclass
class EnsembleFeatures:
    """Feature vector for a single game."""
    net_rating_spread: float = 0.0      # home_net_rtg - away_net_rtg
    player_impact_adj: float = 0.0      # sum of injury adjustments (from home perspective)
    rest_differential: float = 0.0      # home_rest_days - away_rest_days (0 if equal)
    home_court: float = 1.0             # always 1 for home team perspective
    pace_differential: float = 0.0      # home_pace - away_pace
    market_price: float | None = None   # Polymarket implied prob for home team
    clv_residual: float | None = None   # rolling 5-game CLV in probability space

    def to_vector(self) -> list[float]:
        """Convert to feature vector. None → 0 with flag."""
        return [
            self.net_rating_spread,
            self.player_impact_adj,
            self.rest_differential,
            self.home_court,
            self.pace_differential,
            self.market_price if self.market_price is not None else 0.5,
            self.clv_residual if self.clv_residual is not None else 0.0,
        ]

    def has_market_price(self) -> bool:
        return self.market_price is not None


# ---------------------------------------------------------------------------
# Heuristic blend (used before enough training data)
# ---------------------------------------------------------------------------

# Domain-knowledge weights for warm-start
# Net rating and market price are the two strongest signals
HEURISTIC_WEIGHTS = {
    "net_rating_spread": 0.25,
    "player_impact_adj": 0.10,
    "rest_differential": 0.03,
    "home_court": 0.02,
    "pace_differential": 0.00,   # minimal signal for moneyline
    "market_price": 0.55,        # market is very efficient
    "clv_residual": 0.05,
}


def heuristic_probability(features: EnsembleFeatures, base_model_prob: float) -> float:
    """Blend model probability with market price using domain weights.

    When market price is available:
        P = 0.55 * market + 0.45 * model_composite
    When missing:
        P = base_model_prob (no ensemble benefit without market)

    Args:
        features: EnsembleFeatures for this game
        base_model_prob: The 3-layer model probability (from decomposition)

    Returns:
        Blended probability estimate
    """
    if not features.has_market_price():
        return base_model_prob

    market = features.market_price

    # Model composite: base model prob adjusted by player impact info
    # The base_model_prob already includes net rating + situational + info edge,
    # so we don't double-count those features here
    model_weight = 0.45
    market_weight = 0.55

    blended = model_weight * base_model_prob + market_weight * market

    # CLV adjustment: if our recent CLV is positive, trust model more
    if features.clv_residual is not None and features.clv_residual > 0:
        # Shift weight slightly toward model (max +5% shift)
        clv_boost = min(features.clv_residual * 2, 0.05)
        blended = (model_weight + clv_boost) * base_model_prob + (market_weight - clv_boost) * market

    return max(0.05, min(0.95, blended))


# ---------------------------------------------------------------------------
# Online logistic regression
# ---------------------------------------------------------------------------

def _sigmoid(x: float) -> float:
    if x > 500:
        return 1.0
    if x < -500:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


class OnlineEnsemble:
    """Online-learning logistic regression ensemble.

    Warm-starts with heuristic weights, auto-retrains every RETRAIN_EVERY games.
    Uses L2 regularization to prevent overfitting on small samples.
    """

    def __init__(self):
        self.weights: list[float] = [0.0] * 7  # 7 features
        self.bias: float = 0.0
        self.trained: bool = False
        self.training_games: int = 0
        self.games_since_retrain: int = 0
        self.last_retrain: str = ""

        # Try to load persisted model
        self._load()

    def predict(self, features: EnsembleFeatures, base_model_prob: float) -> float:
        """Predict home win probability.

        Uses learned weights if trained on 45+ games, else heuristic blend.
        """
        if not self.trained or self.training_games < MIN_GAMES_TO_TRAIN:
            return heuristic_probability(features, base_model_prob)

        vec = features.to_vector()
        z = self.bias + sum(w * x for w, x in zip(self.weights, vec))
        return max(0.05, min(0.95, _sigmoid(z)))

    def record_outcome(self, features: EnsembleFeatures, home_won: bool):
        """Record a game outcome for future training.

        Stores to Supabase ensemble_training table. Triggers retrain
        if RETRAIN_EVERY new games have accumulated.
        """
        self._store_training_row(features, home_won)
        self.games_since_retrain += 1

        if self.games_since_retrain >= RETRAIN_EVERY:
            self.retrain()

    def retrain(self):
        """Retrain on recent games from Supabase.

        Uses expanding window (max MAX_TRAIN_WINDOW games).
        L2 regularized logistic regression via gradient descent.
        """
        rows = self._fetch_training_data()
        if len(rows) < MIN_GAMES_TO_TRAIN:
            logger.info("Ensemble: only %d games, need %d — keeping heuristic", len(rows), MIN_GAMES_TO_TRAIN)
            return

        # Limit to most recent MAX_TRAIN_WINDOW games
        rows = rows[-MAX_TRAIN_WINDOW:]

        # Extract features and labels
        X = []
        y = []
        for row in rows:
            vec = [
                row.get("net_rating_spread", 0),
                row.get("player_impact_adj", 0),
                row.get("rest_differential", 0),
                row.get("home_court", 1),
                row.get("pace_differential", 0),
                row.get("market_price", 0.5),
                row.get("clv_residual", 0),
            ]
            X.append(vec)
            y.append(1.0 if row.get("home_won") else 0.0)

        # Skip if no market price data in training set
        has_market = sum(1 for row in rows if row.get("market_price") and row["market_price"] != 0.5) / len(rows)
        if has_market < 0.5:
            logger.info("Ensemble: only %.0f%% games have market prices — keeping heuristic", has_market * 100)
            return

        # Train via gradient descent (L2 regularized)
        n_features = 7
        weights = [0.0] * n_features
        bias = 0.0
        lr = 0.01
        l2_reg = 0.5  # C=0.5 equivalent: lambda = 1/(C*n) ≈ regularization
        n = len(X)

        for epoch in range(500):
            grad_w = [0.0] * n_features
            grad_b = 0.0

            for i in range(n):
                z = bias + sum(w * x for w, x in zip(weights, X[i]))
                pred = _sigmoid(z)
                error = pred - y[i]

                for j in range(n_features):
                    grad_w[j] += error * X[i][j] / n
                grad_b += error / n

            # L2 regularization
            for j in range(n_features):
                grad_w[j] += l2_reg * weights[j] / n

            # Update
            for j in range(n_features):
                weights[j] -= lr * grad_w[j]
            bias -= lr * grad_b

        self.weights = weights
        self.bias = bias
        self.trained = True
        self.training_games = n
        self.games_since_retrain = 0
        self.last_retrain = datetime.now(timezone.utc).isoformat()

        self._save()

        # Log feature importance
        feature_names = ["net_rtg", "player_impact", "rest", "home", "pace", "market", "clv"]
        logger.info(
            "Ensemble retrained on %d games. Weights: %s",
            n,
            ", ".join(f"{name}={w:.3f}" for name, w in zip(feature_names, weights)),
        )

    def _store_training_row(self, features: EnsembleFeatures, home_won: bool):
        """Store a training example to Supabase."""
        if not SUPABASE_URL or not SUPABASE_KEY:
            return

        row = {
            "net_rating_spread": round(features.net_rating_spread, 3),
            "player_impact_adj": round(features.player_impact_adj, 4),
            "rest_differential": round(features.rest_differential, 1),
            "home_court": features.home_court,
            "pace_differential": round(features.pace_differential, 1),
            "market_price": features.market_price,
            "clv_residual": features.clv_residual,
            "home_won": home_won,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        try:
            requests.post(
                f"{SUPABASE_URL}/rest/v1/ensemble_training",
                headers={
                    "apikey": SUPABASE_KEY,
                    "Authorization": f"Bearer {SUPABASE_KEY}",
                    "Content-Type": "application/json",
                    "Prefer": "return=minimal",
                },
                json=row,
                timeout=10,
            )
        except Exception as e:
            logger.warning("Failed to store ensemble training row: %s", e)

    def _fetch_training_data(self) -> list[dict]:
        """Fetch training data from Supabase."""
        if not SUPABASE_URL or not SUPABASE_KEY:
            return []

        try:
            resp = requests.get(
                f"{SUPABASE_URL}/rest/v1/ensemble_training",
                headers={
                    "apikey": SUPABASE_KEY,
                    "Authorization": f"Bearer {SUPABASE_KEY}",
                },
                params={
                    "select": "*",
                    "order": "created_at.desc",
                    "limit": str(MAX_TRAIN_WINDOW),
                },
                timeout=15,
            )
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            logger.warning("Failed to fetch ensemble training data: %s", e)
        return []

    def _save(self):
        """Persist model to disk."""
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "weights": self.weights,
            "bias": self.bias,
            "trained": self.trained,
            "training_games": self.training_games,
            "last_retrain": self.last_retrain,
        }
        with open(MODEL_PATH, "w") as f:
            json.dump(data, f, indent=2)

    def _load(self):
        """Load persisted model."""
        if MODEL_PATH.exists():
            try:
                with open(MODEL_PATH) as f:
                    data = json.load(f)
                self.weights = data.get("weights", [0.0] * 7)
                self.bias = data.get("bias", 0.0)
                self.trained = data.get("trained", False)
                self.training_games = data.get("training_games", 0)
                self.last_retrain = data.get("last_retrain", "")
                logger.info("Ensemble loaded: %d games, trained=%s", self.training_games, self.trained)
            except Exception as e:
                logger.warning("Failed to load ensemble model: %s", e)


# Singleton
_ensemble: OnlineEnsemble | None = None


def get_ensemble() -> OnlineEnsemble:
    global _ensemble
    if _ensemble is None:
        _ensemble = OnlineEnsemble()
    return _ensemble


# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------

def extract_features(
    game_data: dict,
    decomposition,  # ProbabilityDecomposition
    market_data: dict | None = None,
) -> EnsembleFeatures:
    """Extract ensemble features from game data and decomposition.

    Called from research_agent after decomposition is computed.
    """
    features = EnsembleFeatures()

    # Feature 1: Net rating spread
    home_rtg = game_data.get("home_net_rtg", {})
    away_rtg = game_data.get("away_net_rtg", {})
    if home_rtg.get("season") and away_rtg.get("season"):
        features.net_rating_spread = (
            home_rtg["season"].get("net_rtg", 0) - away_rtg["season"].get("net_rtg", 0)
        )

    # Feature 2: Player impact sum (already computed in decomposition)
    features.player_impact_adj = decomposition.information_edge.total

    # Feature 3: Rest differential (from situational factors)
    sit = decomposition.situational_adjustment
    # Rest is embedded in the situational adjustment; extract if available
    features.rest_differential = sit.rest_adj * 10  # scale to ~0-3 range

    # Feature 4: Home court (always 1)
    features.home_court = 1.0

    # Feature 5: Pace differential
    if home_rtg.get("season") and away_rtg.get("season"):
        features.pace_differential = (
            home_rtg["season"].get("pace", 100) - away_rtg["season"].get("pace", 100)
        )

    # Feature 6: Market price
    if market_data:
        features.market_price = market_data.get("home_ask") or market_data.get("home_price")

    # Feature 7: CLV residual (populated later if available)
    # Requires CLV data to have accumulated
    try:
        from src.model.clv import check_clv_health
        clv = check_clv_health()
        if clv["sample_size"] >= 5:
            features.clv_residual = clv["rolling_7d_clv_bps"] / 10000  # convert bps to probability
    except Exception:
        pass

    return features
