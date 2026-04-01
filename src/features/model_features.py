from __future__ import annotations

from dataclasses import asdict, dataclass

from src.features.game_features import FeatureSnapshot


@dataclass(frozen=True)
class NBAModelFeatures:
    sport: str
    home_team: str
    away_team: str
    home_court: float
    market_home_probability: float | None
    base_probability: float
    final_probability: float
    home_season_win_pct: float
    away_season_win_pct: float
    home_split_win_pct: float
    away_split_win_pct: float
    home_l10_win_pct: float
    away_l10_win_pct: float
    net_rating_spread: float
    recent_net_rating_spread: float
    pace_differential: float
    rest_differential: float
    home_b2b: float
    away_b2b: float
    travel_zone_differential: float
    player_impact_adjustment: float
    injury_count: float
    clv_residual: float | None

    def to_training_row(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class NHLModelFeatures:
    sport: str
    home_team: str
    away_team: str
    home_ice: float
    market_home_probability: float | None
    base_probability: float
    final_probability: float
    home_points_pct: float
    away_points_pct: float
    home_split_points_pct: float
    away_split_points_pct: float
    home_l10_points_pct: float
    away_l10_points_pct: float
    xgf_pct_spread: float
    gf_ga_ratio_spread: float
    rest_differential: float
    home_b2b: float
    away_b2b: float
    travel_zone_differential: float
    goalie_adjustment: float
    player_impact_adjustment: float
    injury_count: float
    clv_residual: float | None

    def to_training_row(self) -> dict:
        return asdict(self)


def _safe(value: float | None, default: float = 0.0) -> float:
    return default if value is None else float(value)


def _net_rating_spread(home_net: dict, away_net: dict, key: str = "net_rtg") -> float:
    home = (home_net or {}).get(key, 0.0) or 0.0
    away = (away_net or {}).get(key, 0.0) or 0.0
    return round(float(home) - float(away), 4)


def build_nba_model_features(
    feature: FeatureSnapshot,
    *,
    base_probability: float,
    final_probability: float,
) -> NBAModelFeatures:
    home_season = feature.home_net_rtg.get("season", {}) if feature.home_net_rtg else {}
    away_season = feature.away_net_rtg.get("season", {}) if feature.away_net_rtg else {}
    home_l10 = feature.home_net_rtg.get("l10", {}) if feature.home_net_rtg else {}
    away_l10 = feature.away_net_rtg.get("l10", {}) if feature.away_net_rtg else {}

    return NBAModelFeatures(
        sport="NBA",
        home_team=feature.home_team,
        away_team=feature.away_team,
        home_court=1.0,
        market_home_probability=feature.market_home_price,
        base_probability=base_probability,
        final_probability=final_probability,
        home_season_win_pct=_safe(feature.home_win_pct),
        away_season_win_pct=_safe(feature.away_win_pct),
        home_split_win_pct=_safe(feature.home_home_pct, _safe(feature.home_win_pct)),
        away_split_win_pct=_safe(feature.away_away_pct, _safe(feature.away_win_pct)),
        home_l10_win_pct=_safe(feature.home_l10_pct, _safe(feature.home_win_pct)),
        away_l10_win_pct=_safe(feature.away_l10_pct, _safe(feature.away_win_pct)),
        net_rating_spread=_net_rating_spread(home_season, away_season),
        recent_net_rating_spread=_net_rating_spread(home_l10, away_l10),
        pace_differential=_net_rating_spread(home_season, away_season, key="pace"),
        rest_differential=float(feature.rest_differential),
        home_b2b=float(feature.home_is_b2b),
        away_b2b=float(feature.away_is_b2b),
        travel_zone_differential=float(feature.away_travel_zones),
        player_impact_adjustment=float(feature.player_impact_adj),
        injury_count=float(feature.injury_count),
        clv_residual=feature.clv_residual,
    )


def build_nhl_model_features(
    feature: FeatureSnapshot,
    *,
    base_probability: float,
    final_probability: float,
) -> NHLModelFeatures:
    return NHLModelFeatures(
        sport="NHL",
        home_team=feature.home_team,
        away_team=feature.away_team,
        home_ice=1.0,
        market_home_probability=feature.market_home_price,
        base_probability=base_probability,
        final_probability=final_probability,
        home_points_pct=_safe(feature.home_win_pct),
        away_points_pct=_safe(feature.away_win_pct),
        home_split_points_pct=_safe(feature.home_home_pct, _safe(feature.home_win_pct)),
        away_split_points_pct=_safe(feature.away_away_pct, _safe(feature.away_win_pct)),
        home_l10_points_pct=_safe(feature.home_l10_pct, _safe(feature.home_win_pct)),
        away_l10_points_pct=_safe(feature.away_l10_pct, _safe(feature.away_win_pct)),
        xgf_pct_spread=round(_safe(feature.home_xgf_pct) - _safe(feature.away_xgf_pct), 4),
        gf_ga_ratio_spread=round(_safe(feature.home_gf_ga_ratio) - _safe(feature.away_gf_ga_ratio), 4),
        rest_differential=float(feature.rest_differential),
        home_b2b=float(feature.home_is_b2b),
        away_b2b=float(feature.away_is_b2b),
        travel_zone_differential=float(feature.away_travel_zones),
        goalie_adjustment=float(feature.goalie_adjustment),
        player_impact_adjustment=float(feature.player_impact_adj),
        injury_count=float(feature.injury_count),
        clv_residual=feature.clv_residual,
    )
