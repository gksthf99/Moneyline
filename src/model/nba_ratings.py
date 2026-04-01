"""
NBA Adjusted Net Rating model — replacement for win%-based Elo in Layer 1.

Converts team offensive/defensive ratings (per 100 possessions) into
win probability via logistic function with SOS weighting and form blending.

Formula:
    Strength = 0.6 * SOS_Adjusted_NetRtg + 0.4 * Recent_Form_NetRtg
    P(home) = 1 / (1 + 10^(-(home_strength - away_strength + HCA) / K))

Garbage time mitigation: 85% overall NetRtg + 15% clutch NetRtg.
"""

import math
import logging

logger = logging.getLogger(__name__)

# Logistic scaling constant. Controls how aggressively net rating differences
# translate to win probability. Higher K = more moderate probabilities.
# Optimized via grid search over 1,131 NBA games (Oct 2025–Mar 2026):
# K=16.0 + shrinkage=0.65 → Brier 0.2094 (vs 0.2219 old win%-Elo, Δ -0.0125)
# Surface is flat around K=12-22 — model is robust to parameter choice.
NET_RTG_K = 16.0

# Home court advantage in net rating points (~3 pts historically).
# Reduced from raw HCA because we blend in home/away-specific form.
HOME_COURT_NET_RTG = 2.5

# Shrinkage factor — compress toward 0 to reduce overconfidence.
# Optimized: 0.65 (vs 0.70 for old win%-Elo model).
NET_RTG_SHRINKAGE = 0.65

# Garbage time mitigation weights
# Ablation (1,131 games): season-only Brier 0.2064, +clutch = 0.2094.
# Clutch adds noise — disabled until we have per-game clutch data (not season aggregate).
OVERALL_WEIGHT = 1.00
CLUTCH_WEIGHT = 0.00

# Strength blend: season SOS-adjusted vs recent form
# Ablation: season-only (1.0/0.0) = 0.2064, 0.6/0.4 = 0.2094.
# L10 adds noise in backtest (uses end-of-season data), but helps live
# by capturing roster changes. Compromise: 80/20 keeps L10 signal
# without letting it dominate.
SOS_WEIGHT = 0.80
FORM_WEIGHT = 0.20


def garbage_adjusted_net_rtg(
    season_net_rtg: float,
    clutch_net_rtg: float | None = None,
) -> float:
    """Apply garbage-time mitigation by blending with clutch performance.

    Clutch = final 5 min, within 5 pts. Weights: 85% overall + 15% clutch.
    If clutch data unavailable, returns season net rating unchanged.
    """
    if clutch_net_rtg is None:
        return season_net_rtg
    return OVERALL_WEIGHT * season_net_rtg + CLUTCH_WEIGHT * clutch_net_rtg


def team_strength(
    sos_adjusted_net_rtg: float,
    recent_form_net_rtg: float,
) -> float:
    """Compute team strength from SOS-adjusted and recent form ratings.

    Strength = 0.6 * SOS_Adjusted + 0.4 * Recent_Form
    """
    return SOS_WEIGHT * sos_adjusted_net_rtg + FORM_WEIGHT * recent_form_net_rtg


def net_rtg_to_probability(
    home_strength: float,
    away_strength: float,
) -> float:
    """Convert net rating differential to home win probability via logistic.

    Applies shrinkage and home court advantage.
    """
    diff = home_strength - away_strength + HOME_COURT_NET_RTG
    shrunk_diff = diff * NET_RTG_SHRINKAGE
    return 1.0 / (1.0 + 10 ** (-shrunk_diff / NET_RTG_K))


def nba_net_rating_probability(
    home_season: dict,
    away_season: dict,
    home_l10: dict | None = None,
    away_l10: dict | None = None,
    home_clutch: dict | None = None,
    away_clutch: dict | None = None,
    all_teams_season: dict | None = None,
) -> dict:
    """Full NBA base probability from adjusted net ratings.

    Args:
        home_season: {"ortg", "drtg", "net_rtg", "pace", "gp"} from season stats
        away_season: same for away team
        home_l10: L10 stats (optional, for recent form)
        away_l10: L10 stats (optional)
        home_clutch: clutch-time stats (optional, for garbage time mitigation)
        away_clutch: clutch-time stats (optional)
        all_teams_season: all teams' season stats (for SOS calculation)

    Returns:
        {
            "win_prob": float,
            "home_strength": float,
            "away_strength": float,
            "method": "net_rating",
        }
    """
    # Step 1: Garbage-time adjusted net rating
    h_net = garbage_adjusted_net_rtg(
        home_season.get("net_rtg", 0.0),
        home_clutch.get("net_rtg") if home_clutch else None,
    )
    a_net = garbage_adjusted_net_rtg(
        away_season.get("net_rtg", 0.0),
        away_clutch.get("net_rtg") if away_clutch else None,
    )

    # Step 2: SOS adjustment
    # Simple approach: use raw net rating as proxy for SOS-adjusted.
    # A team's net rating already reflects quality of opponents faced over
    # a full season. For a more granular approach, we'd need opponent-by-opponent
    # data. The L10 form accounts for recent schedule difficulty shift.
    h_sos = h_net
    a_sos = a_net

    # If we have all teams' data, compute a basic SOS adjustment:
    # league average net rating is ~0 by definition, so teams facing
    # harder schedules have their rating slightly boosted.
    if all_teams_season:
        league_avg = sum(t.get("net_rtg", 0) for t in all_teams_season.values()) / max(len(all_teams_season), 1)
        # Normalize — league avg should be ~0, but floating point may differ
        h_sos = h_net - league_avg * 0.1  # minor normalization
        a_sos = a_net - league_avg * 0.1

    # Step 3: Recent form from L10 net rating
    h_form = home_l10.get("net_rtg", h_sos) if home_l10 else h_sos
    a_form = away_l10.get("net_rtg", a_sos) if away_l10 else a_sos

    # Step 4: Blend
    h_strength = team_strength(h_sos, h_form)
    a_strength = team_strength(a_sos, a_form)

    # Step 5: Convert to probability
    prob = net_rtg_to_probability(h_strength, a_strength)

    return {
        "win_prob": prob,
        "home_strength": round(h_strength, 2),
        "away_strength": round(a_strength, 2),
        "method": "net_rating",
    }
