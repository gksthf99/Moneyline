"""
Layer 1: Base probability generation.

NBA: Elo rating system (from win% with home/away splits) + logit shrinkage
NHL: Elo rating system (from win% with home/away splits), no shrinkage

Both convert to win probabilities. These are public, reproducible baselines.

Shrinkage factors are empirically validated from a full-season backtest sweep
(2,251 games, Oct 2025 – Mar 2026). They compress overconfident extremes
toward 50% in log-odds space before converting to probability.
"""

import math


# --- Elo System (NBA) ---

# Default K-factor. Higher = more reactive to recent results.
NBA_K = 20
# Home court Elo bonus (~58% expected win rate for equal teams at home)
NBA_HOME_ELO_BONUS = 70
# Base Elo for a .500 team
BASE_ELO = 1500

# Logit shrinkage: multiply Elo difference by this factor before converting
# to probability. <1.0 compresses extremes toward 50%.
# Optimal values from backtest sensitivity sweep (2,251 games):
NBA_SHRINKAGE = 0.70   # NBA overconfident at extremes; 0.70 minimizes Brier
NHL_SHRINKAGE = 1.00   # NHL (with points%) already well-calibrated


def win_pct_to_elo(win_pct: float) -> float:
    """Convert season win percentage to approximate Elo rating.
    .500 = 1500, .750 ≈ 1700, .250 ≈ 1300."""
    if win_pct <= 0:
        win_pct = 0.01
    if win_pct >= 1:
        win_pct = 0.99
    return BASE_ELO + 400 * math.log10(win_pct / (1 - win_pct))


def elo_expected(rating_a: float, rating_b: float) -> float:
    """Expected win probability for team A vs team B (Elo formula)."""
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400))


def nba_base_probability(
    home_win_pct: float,
    away_win_pct: float,
    home_home_pct: float | None = None,
    away_away_pct: float | None = None,
    # Net rating inputs (R1.1 upgrade — used when available)
    home_net_rtg: dict | None = None,
    away_net_rtg: dict | None = None,
) -> float:
    """Generate base win probability for home team.

    When net rating data is provided (home_net_rtg/away_net_rtg dicts with
    'season', 'l10', 'clutch' keys), uses the adjusted net rating model
    (per 100 possessions) which is strictly better than win%-based Elo.

    Falls back to the original win%-to-Elo pipeline when net ratings are
    unavailable (e.g., NHL, backtest without API access).

    Args:
        home_win_pct: Home team's overall season win percentage (fallback)
        away_win_pct: Away team's overall season win percentage (fallback)
        home_home_pct: Home team's win% in HOME games only
        away_away_pct: Away team's win% in AWAY games only
        home_net_rtg: {"season": {...}, "l10": {...}, "clutch": {...}} or None
        away_net_rtg: {"season": {...}, "l10": {...}, "clutch": {...}} or None
    """
    # --- New path: Adjusted Net Rating model ---
    if home_net_rtg and away_net_rtg:
        from src.model.nba_ratings import nba_net_rating_probability
        result = nba_net_rating_probability(
            home_season=home_net_rtg.get("season", {}),
            away_season=away_net_rtg.get("season", {}),
            home_l10=home_net_rtg.get("l10"),
            away_l10=away_net_rtg.get("l10"),
            home_clutch=home_net_rtg.get("clutch"),
            away_clutch=away_net_rtg.get("clutch"),
        )
        return result["win_prob"]

    # --- Legacy path: Win% → Elo ---
    # Blend split with overall to dampen small-sample extremes
    # 70% split + 30% overall — splits are signal but can be noisy
    if home_home_pct is not None:
        h_pct = 0.70 * home_home_pct + 0.30 * home_win_pct
    else:
        h_pct = home_win_pct

    if away_away_pct is not None:
        a_pct = 0.70 * away_away_pct + 0.30 * away_win_pct
    else:
        a_pct = away_win_pct

    # Reduced home bonus (+30) since splits already reflect venue performance
    home_elo = win_pct_to_elo(h_pct) + 30
    away_elo = win_pct_to_elo(a_pct)

    # Shrink Elo difference toward 0 to compress overconfident extremes
    elo_diff = home_elo - away_elo
    shrunk_diff = elo_diff * NBA_SHRINKAGE
    return 1.0 / (1.0 + 10 ** (-shrunk_diff / 400))


# --- NHL Elo System (from win%) ---

# Home ice Elo bonus (~55% expected win rate for equal teams at home)
NHL_HOME_ELO_BONUS = 50

# Home-ice bonus applied after split blending (reduced from NHL_HOME_ELO_BONUS
# because splits already capture venue performance).
HOME_ICE_BONUS = 25

# Suppress home-ice bonus when the away team's blended points% exceeds the
# home team's by more than this threshold.  Based on Lesson 8: 8 of 9 NHL
# misses (Mar 18-21) were home-team favorites in the 51-65% band losing to
# stronger away teams.  The bonus was pulling genuinely close matchups toward
# home when the away team was simply better.
SUPPRESSION_THRESHOLD = 0.05


def nhl_base_probability(
    home_win_pct: float,
    away_win_pct: float,
    home_home_pct: float | None = None,
    away_away_pct: float | None = None,
    home_xgf_pct: float | None = None,
    away_xgf_pct: float | None = None,
    home_gf_ga_ratio: float | None = None,
    away_gf_ga_ratio: float | None = None,
) -> dict:
    """Generate base win probability for NHL home team using Elo.

    Same framework as NBA: convert win% to Elo, add home ice bonus.
    If home/away split win% provided, blends 70% split + 30% overall.

    The home-ice bonus is suppressed when the away team's blended strength
    metric exceeds the home team's by more than SUPPRESSION_THRESHOLD,
    preventing the bonus from flipping the predicted winner in mismatched
    games.

    NHL-specific enhancements:
    - xGF% blend: 70% points% + 30% xGF% — catches teams winning "ugly"
    - Goal differential: minor adjustment for teams outperforming their xG
    NHL_SHRINKAGE = 1.00 (no compression needed — points% is well-calibrated).

    Args:
        home_win_pct: Home team's overall season win percentage (points%)
        away_win_pct: Away team's overall season win percentage (points%)
        home_home_pct: Home team's win% in HOME games only
        away_away_pct: Away team's win% in AWAY games only
        home_xgf_pct: Home team's expected goals for % (0-1, from NST)
        away_xgf_pct: Away team's expected goals for % (0-1, from NST)
        home_gf_ga_ratio: Home team's GF/(GF+GA) ratio
        away_gf_ga_ratio: Away team's GF/(GF+GA) ratio

    Returns:
        dict with keys:
            win_prob (float): Home team win probability
            home_ice_suppressed (bool): True when bonus was suppressed
    """
    if home_home_pct is not None:
        h_pct = 0.70 * home_home_pct + 0.30 * home_win_pct
    else:
        h_pct = home_win_pct

    if away_away_pct is not None:
        a_pct = 0.70 * away_away_pct + 0.30 * away_win_pct
    else:
        a_pct = away_win_pct

    # Blend xGF% into strength metric: 70% points% + 30% xGF%
    # xGF% is a better forward-looking predictor than raw points%
    # but correlated enough that 30% is appropriate until backtest-validated
    NHL_XG_WEIGHT = 0.30
    if home_xgf_pct is not None and away_xgf_pct is not None:
        h_pct = (1 - NHL_XG_WEIGHT) * h_pct + NHL_XG_WEIGHT * home_xgf_pct
        a_pct = (1 - NHL_XG_WEIGHT) * a_pct + NHL_XG_WEIGHT * away_xgf_pct

    # Minor goal differential adjustment: if GF/GA ratio diverges from
    # points%, nudge toward GF/GA (catches unsustainable over/underperformance)
    NHL_GD_WEIGHT = 0.10
    if home_gf_ga_ratio is not None and away_gf_ga_ratio is not None:
        h_pct = (1 - NHL_GD_WEIGHT) * h_pct + NHL_GD_WEIGHT * home_gf_ga_ratio
        a_pct = (1 - NHL_GD_WEIGHT) * a_pct + NHL_GD_WEIGHT * away_gf_ga_ratio

    # Suppress home-ice bonus when the away team is clearly stronger
    # Round to 10 dp to avoid floating-point noise at the boundary
    home_ice_suppressed = round(a_pct - h_pct, 10) > SUPPRESSION_THRESHOLD
    bonus = 0 if home_ice_suppressed else HOME_ICE_BONUS

    home_elo = win_pct_to_elo(h_pct) + bonus
    away_elo = win_pct_to_elo(a_pct)

    elo_diff = home_elo - away_elo
    shrunk_diff = elo_diff * NHL_SHRINKAGE
    win_prob = 1.0 / (1.0 + 10 ** (-shrunk_diff / 400))

    return {"win_prob": win_prob, "home_ice_suppressed": home_ice_suppressed}
