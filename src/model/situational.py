"""
Layer 2: Situational adjustment.

Heuristic overlays with explicit +/- values applied to base probability.
Factors: rest days, back-to-back, travel (cross-timezone).

All adjustments are from the HOME team's perspective.
Positive = home advantage increases. Negative = home advantage decreases.
"""

from datetime import datetime, timedelta
from dataclasses import dataclass


@dataclass
class SituationalFactors:
    """Raw situational inputs for a game."""
    home_days_rest: int  # days since home team's last game
    away_days_rest: int  # days since away team's last game
    home_is_b2b: bool    # home team playing second of back-to-back
    away_is_b2b: bool    # away team playing second of back-to-back
    home_travel_zones: int = 0  # timezone zones crossed by home team in last 24h
    away_travel_zones: int = 0  # timezone zones crossed by away team in last 24h
    # Recent form: L10 win% (0-1). None = no data.
    home_l10_pct: float | None = None
    away_l10_pct: float | None = None
    # Season win% for blending
    home_season_pct: float | None = None
    away_season_pct: float | None = None
    # H2H: home team's win% against this specific opponent
    home_h2h_pct: float | None = None
    h2h_games: int = 0  # number of H2H meetings (for weighting)
    # Goalie adjustment (NHL only): from goalie_adjustment()
    goalie_adj_value: float = 0.0
    goalie_adj_desc: str = ""


@dataclass
class SituationalAdjustment:
    """Breakdown of situational adjustment applied to base probability."""
    rest_adj: float = 0.0       # rest differential adjustment
    b2b_adj: float = 0.0        # back-to-back adjustment
    travel_adj: float = 0.0     # travel fatigue adjustment
    form_adj: float = 0.0       # recent form adjustment
    h2h_adj: float = 0.0        # head-to-head adjustment
    goalie_adj: float = 0.0     # NHL goalie starter/backup adjustment
    total: float = 0.0          # sum of all adjustments
    factors: dict = None        # human-readable explanation

    def __post_init__(self):
        if self.factors is None:
            self.factors = {}


# --- Recent Form ---
# Blended rate: 60% L10 win% + 40% season win%. Difference from season = adjustment.
# Capped at ±5% (NBA) / ±3% (NHL — hockey is more random, form less predictive).
FORM_L10_WEIGHT = 0.60
FORM_SEASON_WEIGHT = 0.40
FORM_CAP = 0.05
NHL_FORM_CAP = 0.03  # reduced from 0.05 — form was biggest contributor to 3/22 misses


def _form_adjustment(
    home_l10: float | None, home_season: float | None,
    away_l10: float | None, away_season: float | None,
    cap: float = FORM_CAP,
) -> tuple[float, str]:
    """Calculate recent form adjustment from home team perspective.

    Args:
        cap: Maximum absolute adjustment. Defaults to FORM_CAP (5%).
             NHL uses NHL_FORM_CAP (3%) — hockey is more random.

    Returns (adjustment, description) or (0.0, "") if no data.
    """
    if home_l10 is None or away_l10 is None:
        return 0.0, ""
    if home_season is None or away_season is None:
        return 0.0, ""

    # Blended rate per team
    home_blended = FORM_L10_WEIGHT * home_l10 + FORM_SEASON_WEIGHT * home_season
    away_blended = FORM_L10_WEIGHT * away_l10 + FORM_SEASON_WEIGHT * away_season

    # How much each team deviates from their season baseline
    home_form_delta = home_blended - home_season
    away_form_delta = away_blended - away_season

    # Net adjustment from home perspective: home improving = positive, away improving = negative
    raw = home_form_delta - away_form_delta
    adj = max(-cap, min(cap, raw))

    if abs(adj) < 0.005:
        return 0.0, ""

    desc = (
        f"Form: home L10 {home_l10:.0%} (season {home_season:.0%}), "
        f"away L10 {away_l10:.0%} (season {away_season:.0%})"
    )
    return adj, desc


# --- H2H Adjustment ---
# If teams have played 2+ times this season, adjust toward H2H win%.
# Weight scales with number of meetings (2 games = light, 4+ = full weight).
# Capped at ±3%.
H2H_CAP = 0.03
H2H_MIN_GAMES = 2


def _h2h_adjustment(
    home_h2h_pct: float | None,
    h2h_games: int,
) -> tuple[float, str]:
    """Calculate H2H adjustment from home team perspective.

    Returns (adjustment, description) or (0.0, "") if insufficient data.
    """
    if home_h2h_pct is None or h2h_games < H2H_MIN_GAMES:
        return 0.0, ""

    # How much H2H deviates from 50% (neutral)
    raw = home_h2h_pct - 0.5

    # Weight by sample size: 2 games = 50%, 3 = 75%, 4+ = 100%
    weight = min(1.0, h2h_games / 4)
    weighted = raw * weight

    adj = max(-H2H_CAP, min(H2H_CAP, weighted))

    if abs(adj) < 0.005:
        return 0.0, ""

    away_pct = 1.0 - home_h2h_pct
    desc = f"H2H: {home_h2h_pct:.0%}-{away_pct:.0%} in {h2h_games} meetings"
    return adj, desc


# --- NBA Adjustments ---
# Source: historical NBA data shows B2B teams lose ~3% more often,
# rest advantages of 2+ days worth ~1.5-2.5%.

NBA_B2B_PENALTY = -0.035        # team on B2B loses ~3.5% win probability
NBA_REST_ADV_PER_DAY = 0.012    # each extra rest day vs opponent: ~1.2%
NBA_REST_ADV_CAP = 0.03         # cap rest advantage at 3%
NBA_TRAVEL_PER_ZONE = -0.01     # each timezone crossed: ~1% penalty
NBA_SITUATIONAL_WEIGHT = 0.25   # backtest-backed dampener for Layer 2


def _scale_adjustment(adjustment: SituationalAdjustment, weight: float, label: str) -> SituationalAdjustment:
    """Scale all adjustment components consistently."""
    adjustment.rest_adj *= weight
    adjustment.b2b_adj *= weight
    adjustment.travel_adj *= weight
    adjustment.form_adj *= weight
    adjustment.h2h_adj *= weight
    adjustment.goalie_adj *= weight
    adjustment.total *= weight
    if adjustment.total != 0:
        adjustment.factors["weight"] = label
    return adjustment


def nba_situational(factors: SituationalFactors) -> SituationalAdjustment:
    """Calculate NBA situational adjustment (from home team perspective).

    Returns adjustment to add to home team's base probability.
    """
    adj = SituationalAdjustment(factors={})

    # Back-to-back
    if factors.home_is_b2b and not factors.away_is_b2b:
        adj.b2b_adj = NBA_B2B_PENALTY
        adj.factors["b2b"] = "Home on B2B, away rested"
    elif factors.away_is_b2b and not factors.home_is_b2b:
        adj.b2b_adj = -NBA_B2B_PENALTY  # positive for home
        adj.factors["b2b"] = "Away on B2B, home rested"
    elif factors.home_is_b2b and factors.away_is_b2b:
        adj.b2b_adj = 0.0
        adj.factors["b2b"] = "Both on B2B — cancels out"

    # Rest differential (only if not already captured by B2B)
    if not (factors.home_is_b2b or factors.away_is_b2b):
        rest_diff = factors.home_days_rest - factors.away_days_rest
        raw_adj = rest_diff * NBA_REST_ADV_PER_DAY
        adj.rest_adj = max(-NBA_REST_ADV_CAP, min(NBA_REST_ADV_CAP, raw_adj))
        if rest_diff != 0:
            adj.factors["rest"] = f"Rest diff: home {factors.home_days_rest}d vs away {factors.away_days_rest}d"

    # Travel
    travel_diff = factors.away_travel_zones - factors.home_travel_zones
    if travel_diff != 0:
        adj.travel_adj = travel_diff * NBA_TRAVEL_PER_ZONE
        adj.factors["travel"] = f"Travel zones: home {factors.home_travel_zones}, away {factors.away_travel_zones}"

    # Recent form
    form_val, form_desc = _form_adjustment(
        factors.home_l10_pct, factors.home_season_pct,
        factors.away_l10_pct, factors.away_season_pct,
    )
    if form_val != 0:
        adj.form_adj = form_val
        adj.factors["form"] = form_desc

    # H2H
    h2h_val, h2h_desc = _h2h_adjustment(factors.home_h2h_pct, factors.h2h_games)
    if h2h_val != 0:
        adj.h2h_adj = h2h_val
        adj.factors["h2h"] = h2h_desc

    adj.total = adj.rest_adj + adj.b2b_adj + adj.travel_adj + adj.form_adj + adj.h2h_adj
    return _scale_adjustment(
        adj,
        NBA_SITUATIONAL_WEIGHT,
        f"NBA situational weight applied: {NBA_SITUATIONAL_WEIGHT:.0%} of raw overlay",
    )


# --- NHL Adjustments ---
# NHL B2B effect is slightly smaller than NBA (~2.5%), travel matters more
# because of more games and cross-country scheduling.
#
# TOTAL_SIT_CAP: Individual factor caps sum to ±18% theoretical max, which
# is absurd for hockey. On 2026-03-22, form+H2H+goalie compounded to +9.6%
# on Stars (Brier=0.5756). This cap prevents compounding regardless of what
# individual signals do. Backtest showed sit adj helps only 0.0009 Brier on
# average — large adjustments are noise, not signal.

NHL_B2B_PENALTY = -0.025
NHL_REST_ADV_PER_DAY = 0.010
NHL_REST_ADV_CAP = 0.025
NHL_TRAVEL_PER_ZONE = -0.012
NHL_TOTAL_SIT_CAP = 0.05  # ±5% max total sit adjustment for NHL


def nhl_situational(factors: SituationalFactors) -> SituationalAdjustment:
    """Calculate NHL situational adjustment (from home team perspective).

    Returns adjustment to add to home team's base probability.
    """
    adj = SituationalAdjustment(factors={})

    # Back-to-back
    if factors.home_is_b2b and not factors.away_is_b2b:
        adj.b2b_adj = NHL_B2B_PENALTY
        adj.factors["b2b"] = "Home on B2B, away rested"
    elif factors.away_is_b2b and not factors.home_is_b2b:
        adj.b2b_adj = -NHL_B2B_PENALTY
        adj.factors["b2b"] = "Away on B2B, home rested"
    elif factors.home_is_b2b and factors.away_is_b2b:
        adj.b2b_adj = 0.0
        adj.factors["b2b"] = "Both on B2B — cancels out"

    # Rest differential
    if not (factors.home_is_b2b or factors.away_is_b2b):
        rest_diff = factors.home_days_rest - factors.away_days_rest
        raw_adj = rest_diff * NHL_REST_ADV_PER_DAY
        adj.rest_adj = max(-NHL_REST_ADV_CAP, min(NHL_REST_ADV_CAP, raw_adj))
        if rest_diff != 0:
            adj.factors["rest"] = f"Rest diff: home {factors.home_days_rest}d vs away {factors.away_days_rest}d"

    # Travel
    travel_diff = factors.away_travel_zones - factors.home_travel_zones
    if travel_diff != 0:
        adj.travel_adj = travel_diff * NHL_TRAVEL_PER_ZONE
        adj.factors["travel"] = f"Travel zones: home {factors.home_travel_zones}, away {factors.away_travel_zones}"

    # Recent form (NHL uses reduced cap — hockey is more random)
    form_val, form_desc = _form_adjustment(
        factors.home_l10_pct, factors.home_season_pct,
        factors.away_l10_pct, factors.away_season_pct,
        cap=NHL_FORM_CAP,
    )
    if form_val != 0:
        adj.form_adj = form_val
        adj.factors["form"] = form_desc

    # H2H
    h2h_val, h2h_desc = _h2h_adjustment(factors.home_h2h_pct, factors.h2h_games)
    if h2h_val != 0:
        adj.h2h_adj = h2h_val
        adj.factors["h2h"] = h2h_desc

    # Goalie (NHL only — pre-computed and injected by research agent)
    if factors.goalie_adj_value != 0:
        adj.goalie_adj = factors.goalie_adj_value
        adj.factors["goalie"] = factors.goalie_adj_desc

    # Total sit adjustment with NHL cap — prevents compounding
    raw_total = adj.rest_adj + adj.b2b_adj + adj.travel_adj + adj.form_adj + adj.h2h_adj + adj.goalie_adj
    adj.total = max(-NHL_TOTAL_SIT_CAP, min(NHL_TOTAL_SIT_CAP, raw_total))

    if adj.total != raw_total:
        adj.factors["cap"] = f"Total capped: {raw_total:+.1%} → {adj.total:+.1%}"

    return adj
