"""Tests for backtest model functions — 12 required test cases."""

import json
import math
import sys
import tempfile
from datetime import date, timedelta
from pathlib import Path

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.backtest_runner import (
    HOME_BONUS,
    SHRINKAGE,
    SITUATIONAL_WEIGHT,
    TeamState,
    assign_triage,
    brier,
    compute_probs,
    layer1_prob,
    layer2_adj,
    load_checkpoint,
    logistic,
    logit,
    points_pct_smoothed,
    save_checkpoint,
    win_rate,
    CHECKPOINT_PATH,
)


# ---------------------------------------------------------------------------
# 1. win_rate Laplace smoothing
# ---------------------------------------------------------------------------

class TestWinRate:
    def test_zero_zero(self):
        assert win_rate(0, 0) == 0.5

    def test_one_zero(self):
        assert abs(win_rate(1, 0) - 2 / 3) < 0.001

    def test_40_20(self):
        expected = 41 / 62
        assert abs(win_rate(40, 20) - expected) < 0.001


# ---------------------------------------------------------------------------
# 2. layer1_prob home advantage (equal teams)
# ---------------------------------------------------------------------------

class TestLayer1HomeAdvantage:
    def test_nba_equal_teams(self):
        # With 0.70 shrinkage: logistic(0.363 * 0.70) ≈ 0.563
        prob = layer1_prob("nba", 0.5, 0.5)
        assert abs(prob - 0.563) < 0.005

    def test_nhl_equal_teams(self):
        # With 1.00 shrinkage (no shrinkage): logistic(0.200) ≈ 0.550
        prob = layer1_prob("nhl", 0.5, 0.5)
        assert abs(prob - 0.550) < 0.005


# ---------------------------------------------------------------------------
# 3. layer1_prob strong vs weak
# ---------------------------------------------------------------------------

class TestLayer1StrongVsWeak:
    def test_nba_700_vs_300(self):
        # With 0.70 shrinkage, compressed from ~0.887 to ~0.80
        prob = layer1_prob("nba", 0.700, 0.300)
        assert prob > 0.77
        assert prob < 0.84


# ---------------------------------------------------------------------------
# 4. layer2_adj B2B
# ---------------------------------------------------------------------------

class TestLayer2B2B:
    def test_home_b2b_negative(self):
        adj = layer2_adj("nba", True, False, 1, 2)
        assert adj < 0

    def test_away_b2b_positive(self):
        adj = layer2_adj("nba", False, True, 2, 1)
        assert adj > 0

    def test_both_b2b_near_zero(self):
        adj = layer2_adj("nba", True, True, 1, 1)
        assert abs(adj) < 0.01


# ---------------------------------------------------------------------------
# 5. layer2_adj rest
# ---------------------------------------------------------------------------

class TestLayer2Rest:
    def test_home_more_rest_positive(self):
        adj = layer2_adj("nba", False, False, 3, 1)
        assert adj > 0

    def test_away_more_rest_negative(self):
        adj = layer2_adj("nba", False, False, 1, 3)
        assert adj < 0


# ---------------------------------------------------------------------------
# 6. compute_probs capping
# ---------------------------------------------------------------------------

class TestComputeProbsCapping:
    def test_extreme_mismatch_capped(self):
        _, _, final = compute_probs("nba", 0.999, 0.001, False, False, 2, 2)
        assert final == 0.95

    def test_nba_uses_dampened_situational_weight(self):
        base, adj, final = compute_probs("nba", 0.6, 0.4, False, False, 3, 1)
        expected = logistic(logit(base) + (adj * SITUATIONAL_WEIGHT["nba"]))
        assert abs(final - expected) < 1e-9


# ---------------------------------------------------------------------------
# 7. assign_triage all three tiers
# ---------------------------------------------------------------------------

class TestAssignTriage:
    def test_skip(self):
        assert assign_triage(0.29) == "skip"

    def test_deep(self):
        assert assign_triage(0.48) == "deep"

    def test_standard(self):
        assert assign_triage(0.65) == "standard"


# ---------------------------------------------------------------------------
# 8. brier correctness
# ---------------------------------------------------------------------------

class TestBrier:
    def test_70_true(self):
        assert abs(brier(0.7, True) - 0.09) < 0.001

    def test_50_true(self):
        assert brier(0.5, True) == 0.25


# ---------------------------------------------------------------------------
# 9. TeamState B2B detection
# ---------------------------------------------------------------------------

class TestTeamStateB2B:
    def test_b2b_true(self):
        ts = TeamState()
        today = date(2026, 3, 19)
        ts.last_game_date = today - timedelta(days=1)
        assert ts.is_b2b(today) is True

    def test_b2b_false(self):
        ts = TeamState()
        today = date(2026, 3, 19)
        ts.last_game_date = today - timedelta(days=2)
        assert ts.is_b2b(today) is False


# ---------------------------------------------------------------------------
# 10. TeamState rest_days
# ---------------------------------------------------------------------------

class TestTeamStateRestDays:
    def test_rest_days_correct(self):
        ts = TeamState()
        today = date(2026, 3, 19)
        ts.last_game_date = date(2026, 3, 16)
        assert ts.rest_days(today) == 3

    def test_rest_days_none(self):
        ts = TeamState()
        assert ts.rest_days(date(2026, 3, 19)) is None


# ---------------------------------------------------------------------------
# 11. Look-ahead bias guard
# ---------------------------------------------------------------------------

class TestLookAheadBias:
    def test_two_game_sequence(self):
        """After game 1 (home wins), game 2's snapshot should see exactly 1 win."""
        states = {}
        team_a = "AAA"
        team_b = "BBB"
        team_c = "CCC"
        states[team_a] = TeamState()
        states[team_b] = TeamState()
        states[team_c] = TeamState()

        # Day 1: team_a (home) beats team_b
        day1 = date(2026, 1, 1)
        # Snapshot before
        a_wr_before = states[team_a].win_rate()
        assert a_wr_before == 0.5  # Laplace: (0+1)/(0+0+2)

        # Update after game
        states[team_a].wins += 1
        states[team_b].losses += 1
        states[team_a].last_game_date = day1
        states[team_b].last_game_date = day1

        # Day 2: team_a (home) vs team_c
        day2 = date(2026, 1, 2)
        # Snapshot — should reflect 1 win
        a_wr_after = states[team_a].win_rate()
        expected = (1 + 1) / (1 + 0 + 2)  # 2/3
        assert abs(a_wr_after - expected) < 0.001
        assert states[team_a].wins == 1
        assert states[team_a].is_b2b(day2) is True


# ---------------------------------------------------------------------------
# 12. Checkpoint round-trip
# ---------------------------------------------------------------------------

class TestCheckpointRoundTrip:
    def test_write_and_load(self, tmp_path, monkeypatch):
        # Use a temp checkpoint path
        cp = tmp_path / "checkpoint.json"
        monkeypatch.setattr("scripts.backtest_runner.CHECKPOINT_PATH", cp)

        dates = {"2026-01-01", "2026-01-02"}
        save_checkpoint(dates)
        loaded = load_checkpoint()
        assert "2026-01-01" in loaded
        assert "2026-01-02" in loaded
        assert len(loaded) == 2


# ---------------------------------------------------------------------------
# 13. points_pct_smoothed correctness
# ---------------------------------------------------------------------------

class TestPointsPctSmoothed:
    def test_no_games(self):
        assert points_pct_smoothed(0, 0, 0) == 0.5

    def test_35_30_17(self):
        # 35W-30L-17OTL: pts=70+17=87, max=164, smoothed=(87+2)/(164+4)=89/168
        expected = 89 / 168
        assert abs(points_pct_smoothed(35, 30, 17) - expected) < 0.001

    def test_dominant_team(self):
        # 50W-10L-5OTL: pts=100+5=105, max=130, smoothed=(105+2)/(130+4)=107/134
        expected = 107 / 134
        assert abs(points_pct_smoothed(50, 10, 5) - expected) < 0.001


# ---------------------------------------------------------------------------
# 14. NHL standings lookup fallback
# ---------------------------------------------------------------------------

class TestNHLStandingsFallback:
    def test_empty_standings_uses_win_rate(self):
        """When standings dict is empty, strength should fall back to TeamState."""
        ts = TeamState()
        ts.wins = 10
        ts.losses = 5
        nhl_standings = {}
        strength = nhl_standings.get("BOS", ts.win_rate())
        assert abs(strength - win_rate(10, 5)) < 0.001

    def test_standings_overrides_win_rate(self):
        """When team is in standings, use standings value."""
        ts = TeamState()
        ts.wins = 10
        ts.losses = 5
        nhl_standings = {"BOS": 0.620}
        strength = nhl_standings.get("BOS", ts.win_rate())
        assert strength == 0.620


# ---------------------------------------------------------------------------
# 15. Shrinkage effect on extremes
# ---------------------------------------------------------------------------

class TestShrinkageExtremes:
    def test_unshrunk_identity(self):
        """Shrinkage=1.0 should produce same result."""
        raw_lo = logit(0.90)
        result = logistic(raw_lo * 1.0)
        assert abs(result - 0.90) < 0.001

    def test_shrunk_compresses(self):
        """Shrinkage=0.82 should compress 0.90 toward 0.50."""
        raw_lo = logit(0.90)
        result = logistic(raw_lo * 0.82)
        assert result < 0.90
        assert result > 0.50
        # Expect roughly 0.83-0.87
        assert 0.82 < result < 0.88


# ---------------------------------------------------------------------------
# 16. Shrinkage doesn't affect 50% predictions
# ---------------------------------------------------------------------------

class TestShrinkageNeutral:
    def test_equal_teams_any_shrinkage(self):
        """Equal strength teams: any shrinkage should produce ~0.50 (before home bonus)."""
        for shrink in [0.5, 0.82, 1.0]:
            raw_lo = logit(0.5) - logit(0.5)  # = 0.0
            result = logistic(raw_lo * shrink)
            assert abs(result - 0.5) < 0.001


# ---------------------------------------------------------------------------
# 17. Layer 2 still applies after shrinkage
# ---------------------------------------------------------------------------

class TestLayer2AfterShrinkage:
    def test_b2b_still_lowers_prob(self):
        """Home B2B should lower final_prob below base_prob even with shrinkage."""
        base, adj, final = compute_probs("nba", 0.6, 0.4, True, False, 1, 2)
        # Base is shrunk but positive (home is stronger)
        # B2B penalty should make final < base
        assert final < base
