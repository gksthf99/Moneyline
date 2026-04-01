from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReplacementModelSpec:
    sport: str
    model_name: str
    target: str
    estimator: str
    feature_names: tuple[str, ...]
    rationale: tuple[str, ...]
    evaluation_metrics: tuple[str, ...] = (
        "brier_score",
        "log_loss",
        "calibration_by_bucket",
        "favorite_underdog_split",
        "edge_precision",
    )


NBA_REPLACEMENT_MODEL = ReplacementModelSpec(
    sport="NBA",
    model_name="nba_point_in_time_logistic_v1",
    target="home_win",
    estimator="regularized_logistic_regression",
    feature_names=(
        "home_court",
        "market_home_probability",
        "base_probability",
        "final_probability",
        "home_season_win_pct",
        "away_season_win_pct",
        "home_split_win_pct",
        "away_split_win_pct",
        "home_l10_win_pct",
        "away_l10_win_pct",
        "net_rating_spread",
        "recent_net_rating_spread",
        "pace_differential",
        "rest_differential",
        "home_b2b",
        "away_b2b",
        "travel_zone_differential",
        "player_impact_adjustment",
        "injury_count",
        "clv_residual",
    ),
    rationale=(
        "Net rating should remain the anchor for NBA team strength.",
        "Recent form should be learned as a weight, not injected as a large heuristic overlay.",
        "Measured player-impact and market context should calibrate the baseline rather than replace it.",
    ),
)


NHL_REPLACEMENT_MODEL = ReplacementModelSpec(
    sport="NHL",
    model_name="nhl_point_in_time_logistic_v1",
    target="home_win",
    estimator="regularized_logistic_regression",
    feature_names=(
        "home_ice",
        "market_home_probability",
        "base_probability",
        "final_probability",
        "home_points_pct",
        "away_points_pct",
        "home_split_points_pct",
        "away_split_points_pct",
        "home_l10_points_pct",
        "away_l10_points_pct",
        "xgf_pct_spread",
        "gf_ga_ratio_spread",
        "rest_differential",
        "home_b2b",
        "away_b2b",
        "travel_zone_differential",
        "goalie_adjustment",
        "player_impact_adjustment",
        "injury_count",
        "clv_residual",
    ),
    rationale=(
        "NHL needs a conservative strength model with goalie state treated explicitly.",
        "Correlated heuristic overlays should be replaced by learned coefficients on a smaller feature set.",
        "Market probability is a useful calibration feature in hockey because of higher outcome variance.",
    ),
)


TENNIS_REPLACEMENT_MODEL = ReplacementModelSpec(
    sport="TENNIS",
    model_name="tennis_point_in_time_logistic_v1",
    target="player_a_win",
    estimator="regularized_logistic_regression",
    feature_names=(
        "tour",
        "market_player_a_price",
        "player_a_rating",
        "player_b_rating",
        "player_a_surface_rating",
        "player_b_surface_rating",
        "player_a_recent_form",
        "player_b_recent_form",
        "player_a_hold_pct",
        "player_b_hold_pct",
        "player_a_break_pct",
        "player_b_break_pct",
        "player_a_rest_days",
        "player_b_rest_days",
        "player_a_last_match_minutes",
        "player_b_last_match_minutes",
        "player_a_travel_zones",
        "player_b_travel_zones",
        "player_a_injury_risk",
        "player_b_injury_risk",
        "h2h_player_a_win_pct",
        "h2h_sample",
    ),
    rationale=(
        "Tennis should be modeled as a player-strength problem, not a team-sport overlay problem.",
        "Surface fit, fatigue, and injury/withdrawal risk matter more than generic situational heuristics.",
        "Market probability is a useful calibration feature but should not dominate the player model.",
    ),
)


def get_replacement_model_spec(sport: str) -> ReplacementModelSpec:
    sport_key = sport.upper()
    if sport_key == "NBA":
        return NBA_REPLACEMENT_MODEL
    if sport_key == "NHL":
        return NHL_REPLACEMENT_MODEL
    if sport_key == "TENNIS":
        return TENNIS_REPLACEMENT_MODEL
    raise ValueError(f"Unsupported sport for replacement model spec: {sport}")
