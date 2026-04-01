from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class FeatureSnapshot:
    """Canonical point-in-time model inputs derived from raw game data."""

    sport: str
    home_team: str
    away_team: str
    home_win_pct: float | None = None
    away_win_pct: float | None = None
    home_home_pct: float | None = None
    away_away_pct: float | None = None
    home_l10_pct: float | None = None
    away_l10_pct: float | None = None
    home_net_rtg: dict = field(default_factory=dict)
    away_net_rtg: dict = field(default_factory=dict)
    home_xgf_pct: float | None = None
    away_xgf_pct: float | None = None
    home_gf_ga_ratio: float | None = None
    away_gf_ga_ratio: float | None = None
    rest_differential: float = 0.0
    home_days_rest: int = 0
    away_days_rest: int = 0
    home_is_b2b: bool = False
    away_is_b2b: bool = False
    away_travel_zones: int = 0
    market_home_price: float | None = None
    market_away_price: float | None = None
    market_liquidity: float | None = None
    goalie_adjustment: float = 0.0
    player_impact_adj: float = 0.0
    clv_residual: float | None = None
    injury_count: int = 0
    freshness_disagreements: tuple[str, ...] = ()


def _parse_split_pct(record_str: str) -> float | None:
    if not record_str:
        return None
    parts = record_str.split("-")
    try:
        wins = int(parts[0])
        total = sum(int(part) for part in parts)
        return wins / total if total > 0 else None
    except (TypeError, ValueError):
        return None


def _parse_nba_l10_pct(form: dict) -> float | None:
    record = (form or {}).get("record")
    return _parse_split_pct(record) if record else None


def build_feature_snapshot(game: dict, game_data: dict, decomposition=None) -> FeatureSnapshot:
    sport = (game.get("sport") or game_data.get("sport") or "").upper()
    market = game_data.get("market") or {}
    situational_inputs = game_data.get("_situational_inputs") or {}

    home_xgf_raw = game_data.get("home_advanced", {}).get("xGF%")
    away_xgf_raw = game_data.get("away_advanced", {}).get("xGF%")
    home_xgf = float(home_xgf_raw) / 100.0 if home_xgf_raw else None
    away_xgf = float(away_xgf_raw) / 100.0 if away_xgf_raw else None

    home_gf = game_data.get("home_gf", 0) or 0
    home_ga = game_data.get("home_ga", 0) or 0
    away_gf = game_data.get("away_gf", 0) or 0
    away_ga = game_data.get("away_ga", 0) or 0
    goalie_adjustment = (game_data.get("goalie_adjustment") or {}).get("value", 0.0) or 0.0
    player_impact_adj = decomposition.information_edge.total if decomposition else 0.0
    clv_residual = game_data.get("_clv_residual")
    home_days_rest = situational_inputs.get("home_days_rest", 0) or 0
    away_days_rest = situational_inputs.get("away_days_rest", 0) or 0

    return FeatureSnapshot(
        sport=sport,
        home_team=game.get("home_team", ""),
        away_team=game.get("away_team", ""),
        home_win_pct=game_data.get("home_win_pct"),
        away_win_pct=game_data.get("away_win_pct"),
        home_home_pct=_parse_split_pct(game_data.get("home_home_record", "")),
        away_away_pct=_parse_split_pct(game_data.get("away_away_record", "")),
        home_l10_pct=_parse_nba_l10_pct(game_data.get("home_form", {})) if sport == "NBA" else _parse_split_pct(game_data.get("home_l10", "")),
        away_l10_pct=_parse_nba_l10_pct(game_data.get("away_form", {})) if sport == "NBA" else _parse_split_pct(game_data.get("away_l10", "")),
        home_net_rtg=game_data.get("home_net_rtg", {}) or {},
        away_net_rtg=game_data.get("away_net_rtg", {}) or {},
        home_xgf_pct=home_xgf,
        away_xgf_pct=away_xgf,
        home_gf_ga_ratio=(home_gf / (home_gf + home_ga)) if (home_gf + home_ga) > 0 else None,
        away_gf_ga_ratio=(away_gf / (away_gf + away_ga)) if (away_gf + away_ga) > 0 else None,
        rest_differential=home_days_rest - away_days_rest,
        home_days_rest=home_days_rest,
        away_days_rest=away_days_rest,
        home_is_b2b=bool(situational_inputs.get("home_is_b2b", False)),
        away_is_b2b=bool(situational_inputs.get("away_is_b2b", False)),
        away_travel_zones=int(situational_inputs.get("away_travel_zones", 0) or 0),
        market_home_price=market.get("home_ask") or market.get("home_price"),
        market_away_price=market.get("away_ask") or market.get("away_price"),
        market_liquidity=market.get("liquidity"),
        goalie_adjustment=float(goalie_adjustment),
        player_impact_adj=float(player_impact_adj),
        clv_residual=clv_residual,
        injury_count=len(game_data.get("injuries", [])),
        freshness_disagreements=tuple(game_data.get("disagreements", [])),
    )
