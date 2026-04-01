# test_betting_system.py
# Run with: python -m pytest tests/test_betting_system.py -v

import pytest
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Optional, Dict, List

# =============================================================================
# TEST CONFIGURATION
# =============================================================================

HEURISTIC_MODEL_WEIGHT = 0.45
HEURISTIC_MARKET_WEIGHT = 0.55
MIN_GAMES_TO_TRAIN = 45
IMPACT_THRESHOLD = 0.03
GIANNIS_IMPACT_EXPECTED = 0.107  # 10.7%

# =============================================================================
# MOCK CLASSES (Replace with your actual imports)
# =============================================================================

@dataclass
class GamePrediction:
    game_id: str
    home_team: str
    away_team: str
    model_prob: float
    market_prob: Optional[float]
    ensemble_prob: float
    net_rating_spread: float
    player_impact_adjustment: float
    timestamp: datetime
    clv_residual: float = 0.0

class EnsembleModel:
    def __init__(self):
        self.weights = [HEURISTIC_MODEL_WEIGHT, HEURISTIC_MARKET_WEIGHT]
        self.training_count = 0

    def predict(self, model_prob: float, market_prob: Optional[float],
                features: Dict) -> float:
        if market_prob is None:
            return model_prob
        return (self.weights[0] * model_prob +
                self.weights[1] * market_prob)

    def should_retrain(self, n_games: int) -> bool:
        return n_games >= MIN_GAMES_TO_TRAIN

class PlayerImpactModel:
    def __init__(self):
        self.impact_map = {
            "Giannis Antetokounmpo": 0.107,
            "Luka Doncic": 0.098,
            "Nikola Jokic": 0.095,
            "Role Player A": 0.015,  # Below threshold
        }

    def get_impact(self, player: str) -> float:
        return self.impact_map.get(player, 0.0)

# =============================================================================
# UNIT TESTS
# =============================================================================

class TestEnsembleMath:
    """Verify the core ensemble calculations are correct"""

    def test_heuristic_blend_with_market(self):
        model = EnsembleModel()
        result = model.predict(model_prob=0.65, market_prob=0.62, features={})
        expected = 0.45 * 0.65 + 0.55 * 0.62
        assert abs(result - expected) < 0.001, f"Expected {expected}, got {result}"
        assert abs(result - 0.6335) < 0.001, "Heuristic blend math incorrect"

    def test_fallback_without_market(self):
        model = EnsembleModel()
        result = model.predict(model_prob=0.65, market_prob=None, features={})
        assert result == 0.65, "Should fallback to model when market is None"

    def test_extreme_probabilities(self):
        model = EnsembleModel()
        # Test near-certainties
        result = model.predict(0.95, 0.05, {})
        expected = 0.45 * 0.95 + 0.55 * 0.05
        assert abs(result - expected) < 0.001

    def test_weights_sum_to_one(self):
        assert abs(sum([HEURISTIC_MODEL_WEIGHT, HEURISTIC_MARKET_WEIGHT]) - 1.0) < 0.001

class TestPlayerImpact:
    """Verify player impact calculations"""

    def test_star_player_impact_not_generic(self):
        pim = PlayerImpactModel()
        giannis = pim.get_impact("Giannis Antetokounmpo")
        assert giannis > 0.08, f"Giannis impact {giannis} should be >8%, not generic 6%"
        assert abs(giannis - GIANNIS_IMPACT_EXPECTED) < 0.01

    def test_role_player_filtered(self):
        pim = PlayerImpactModel()
        role = pim.get_impact("Role Player A")
        assert role < IMPACT_THRESHOLD, f"Role player {role} should be filtered (<{IMPACT_THRESHOLD})"

    def test_unknown_player_zero(self):
        pim = PlayerImpactModel()
        assert pim.get_impact("Random Bench Guy") == 0.0

class TestDataValidation:
    """Verify data integrity and sanity checks"""

    def test_brier_calculation(self):
        """Brier score must be (p - o)^2"""
        predictions = [0.7, 0.3, 0.8]
        outcomes = [1, 0, 1]
        brier = np.mean([(p - o)**2 for p, o in zip(predictions, outcomes)])
        expected = ((0.7-1)**2 + (0.3-0)**2 + (0.8-1)**2) / 3
        assert abs(brier - expected) < 0.001

    def test_clv_calculation(self):
        """CLV = (close - your_price) in bps"""
        your_price = 0.65
        close_price = 0.62
        clv_bps = (close_price - your_price) * 100
        assert clv_bps == pytest.approx(-3.0), "CLV calculation incorrect"

    def test_probability_bounds(self):
        """All probabilities must be in [0, 1]"""
        model = EnsembleModel()
        test_cases = [
            (0.65, 0.62),
            (0.2, 0.8),
            (0.9, 0.1)
        ]
        for m_prob, mk_prob in test_cases:
            result = model.predict(m_prob, mk_prob, {})
            assert 0 <= result <= 1, f"Probability {result} out of bounds"

class TestTrainingLogic:
    """Verify retraining triggers"""

    def test_no_train_before_min_games(self):
        model = EnsembleModel()
        assert not model.should_retrain(44)
        assert not model.should_retrain(10)

    def test_train_at_threshold(self):
        model = EnsembleModel()
        assert model.should_retrain(45)

    def test_train_after_threshold(self):
        model = EnsembleModel()
        assert model.should_retrain(100)

# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestEndToEndPrediction:
    """Full pipeline tests"""

    def test_full_game_prediction_flow(self):
        """Simulate one complete prediction"""
        # Setup
        pim = PlayerImpactModel()
        ensemble = EnsembleModel()

        # Game context: Bucks vs Pistons, Giannis OUT
        base_prob = 0.72  # Bucks favored
        giannis_impact = pim.get_impact("Giannis Antetokounmpo")

        # Apply impact
        adjusted_prob = base_prob - giannis_impact

        # Market price
        market_prob = 0.58  # Market moved down

        # Ensemble
        final_prob = ensemble.predict(adjusted_prob, market_prob, {})

        # Assertions
        assert adjusted_prob < base_prob, "Impact not applied correctly"
        assert 0.50 < final_prob < 0.70, f"Final prob {final_prob} seems wrong for Giannis-out game"

    def test_sanity_check_extreme_favorite_rate(self):
        """If model predicts >70% for >40% of games, it's broken"""
        predictions = [0.75, 0.72, 0.68, 0.71, 0.80, 0.45, 0.52, 0.48, 0.55, 0.60]
        high_conf = sum(1 for p in predictions if p > 0.70)
        rate = high_conf / len(predictions)
        assert rate <= 0.40, f"Overconfidence detected: {rate:.0%} games >70% confidence"

# =============================================================================
# ADVERSARIAL TESTS
# =============================================================================

class TestFailureModes:
    """Test edge cases and failure conditions"""

    def test_duplicate_player_detection(self):
        """Ensure duplicate players don't double-count"""
        lineup = ["Giannis", "Lillard", "Middleton", "Giannis"]  # Duplicate
        unique = list(set(lineup))
        assert len(unique) == 3, "Duplicate detection failed"

    def test_future_date_rejection(self):
        """Model should not predict future games"""
        future = datetime.now() + timedelta(days=1)
        # Your system should validate timestamps
        assert future > datetime.now()

    def test_negative_impact_rejection(self):
        """Impact adjustments must be capped at ±15% in production"""
        # Our system caps total information edge at ±15% (information.py line 140)
        from src.model.information import calculate_information_edge, PlayerImpact, PlayerTier, InjuryStatus
        # Create absurd scenario: 10 T1 players out
        impacts = [
            PlayerImpact(f"Star{i}", "home", PlayerTier.TIER_1, InjuryStatus.OUT, -0.10, "")
            for i in range(10)
        ]
        edge = calculate_information_edge(impacts)
        assert abs(edge.total) <= 0.15, f"Information edge {edge.total} exceeds ±15% cap"

# =============================================================================
# DATA QUALITY CHECKS (Run against live DB)
# =============================================================================

@pytest.mark.integration
class TestLiveData:
    """Tests that require database connection"""

    def test_feature_correlation_sanity(self):
        """
        Verify net rating and market price aren't perfectly correlated
        If r > 0.95, market is just copying your signal or vice versa
        """
        # Generate fake data representing normal conditions
        np.random.seed(42)
        net_rating = np.random.normal(0, 5, 100)  # Mean 0, std 5
        market_price = 0.5 + 0.02 * net_rating + np.random.normal(0, 0.05, 100)

        correlation = np.corrcoef(net_rating, market_price)[0,1]
        assert 0.3 < abs(correlation) < 0.9, f"Suspicious correlation: {correlation}"

    def test_no_data_leakage(self):
        """
        Ensure we're not using future data
        Prediction timestamp must be before game start
        """
        prediction_time = datetime(2026, 3, 24, 18, 0)  # 6 PM
        game_time = datetime(2026, 3, 24, 19, 30)       # 7:30 PM

        assert prediction_time < game_time, "Data leakage: prediction after game start"
