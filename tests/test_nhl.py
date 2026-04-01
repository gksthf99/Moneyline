"""Tests for NHL home-ice bonus suppression in baseline model."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model.baseline import (
    HOME_ICE_BONUS,
    SUPPRESSION_THRESHOLD,
    nhl_base_probability,
)


class TestHomeIceSuppression:
    """Verify the home-ice bonus is suppressed when the away team's blended
    strength exceeds the home team's by more than SUPPRESSION_THRESHOLD."""

    def test_normal_application(self):
        """Home-ice bonus applied when home team is stronger."""
        result = nhl_base_probability(
            home_win_pct=0.60,
            away_win_pct=0.50,
        )
        assert isinstance(result, dict)
        assert "win_prob" in result
        assert "home_ice_suppressed" in result
        assert result["home_ice_suppressed"] is False
        # Home team stronger + bonus → should be well above 50%
        assert result["win_prob"] > 0.55

    def test_suppression_above_threshold(self):
        """Bonus suppressed when away team clearly stronger (gap > 5%)."""
        # Away 65% vs Home 50% → gap = 0.15, well above 0.05
        result = nhl_base_probability(
            home_win_pct=0.50,
            away_win_pct=0.65,
        )
        assert result["home_ice_suppressed"] is True

        # Without suppression the bonus would push home upward; verify that
        # the suppressed probability is lower than a non-suppressed version.
        # We can compare against a scenario where teams are equal (no
        # suppression fires) to confirm the bonus isn't being applied.
        result_no_gap = nhl_base_probability(
            home_win_pct=0.50,
            away_win_pct=0.50,
        )
        # Equal teams with bonus → home > 50%. Away-dominant suppressed → home < 50%.
        assert result_no_gap["home_ice_suppressed"] is False
        assert result_no_gap["win_prob"] > 0.50
        assert result["win_prob"] < 0.50

    def test_suppression_at_exact_threshold(self):
        """Bonus NOT suppressed when gap equals exactly SUPPRESSION_THRESHOLD.

        The condition is strict greater-than (>), so the boundary case
        where away - home == threshold should keep the bonus active.
        """
        home = 0.50
        away = home + SUPPRESSION_THRESHOLD  # exactly at boundary

        result = nhl_base_probability(
            home_win_pct=home,
            away_win_pct=away,
        )
        assert result["home_ice_suppressed"] is False

    def test_no_suppression_away_worse(self):
        """Bonus applied normally when away team is worse than home team."""
        result = nhl_base_probability(
            home_win_pct=0.65,
            away_win_pct=0.40,
        )
        assert result["home_ice_suppressed"] is False
        # Large home advantage + bonus → strong favorite
        assert result["win_prob"] > 0.70

    def test_suppression_with_xgf_blend(self):
        """Suppression uses the fully blended metric (incl. xGF%)."""
        # Raw points%: home 0.55, away 0.55 → equal, no suppression.
        # But xGF%: home 0.45, away 0.60 → after 30% xGF blend,
        # h_pct = 0.70*0.55 + 0.30*0.45 = 0.52
        # a_pct = 0.70*0.55 + 0.30*0.60 = 0.565
        # gap = 0.565 - 0.52 = 0.045 → below threshold, no suppression
        result_below = nhl_base_probability(
            home_win_pct=0.55,
            away_win_pct=0.55,
            home_xgf_pct=0.45,
            away_xgf_pct=0.60,
        )
        assert result_below["home_ice_suppressed"] is False

        # Push xGF gap wider so blended gap > 0.05
        # h_pct = 0.70*0.55 + 0.30*0.40 = 0.505
        # a_pct = 0.70*0.55 + 0.30*0.65 = 0.58
        # gap = 0.58 - 0.505 = 0.075 → above threshold
        result_above = nhl_base_probability(
            home_win_pct=0.55,
            away_win_pct=0.55,
            home_xgf_pct=0.40,
            away_xgf_pct=0.65,
        )
        assert result_above["home_ice_suppressed"] is True

    def test_return_type(self):
        """Return value is always a dict with the expected keys."""
        result = nhl_base_probability(0.50, 0.50)
        assert isinstance(result, dict)
        assert isinstance(result["win_prob"], float)
        assert isinstance(result["home_ice_suppressed"], bool)
        assert 0.0 < result["win_prob"] < 1.0

    def test_constants_are_expected_values(self):
        """Guard against accidental constant changes."""
        assert HOME_ICE_BONUS == 25
        assert SUPPRESSION_THRESHOLD == 0.05
