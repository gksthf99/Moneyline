from __future__ import annotations

from dataclasses import asdict, dataclass

from src.domain.tennis_match import TennisMatchSnapshot


@dataclass(frozen=True)
class TennisFeatureSnapshot:
    sport: str
    player_a: str
    player_b: str
    tour: str
    tournament: str
    round_name: str
    surface: str
    best_of: int
    indoor: bool
    market_player_a_price: float | None = None
    market_player_b_price: float | None = None
    player_a_rating: float = 1500.0
    player_b_rating: float = 1500.0
    player_a_surface_rating: float = 1500.0
    player_b_surface_rating: float = 1500.0
    player_a_recent_form: float = 0.5
    player_b_recent_form: float = 0.5
    player_a_hold_pct: float = 0.75
    player_b_hold_pct: float = 0.75
    player_a_break_pct: float = 0.22
    player_b_break_pct: float = 0.22
    player_a_rest_days: int = 1
    player_b_rest_days: int = 1
    player_a_last_match_minutes: int = 0
    player_b_last_match_minutes: int = 0
    player_a_travel_zones: int = 0
    player_b_travel_zones: int = 0
    player_a_injury_risk: float = 0.0
    player_b_injury_risk: float = 0.0
    h2h_player_a_win_pct: float | None = None
    h2h_sample: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def build_tennis_feature_snapshot(snapshot: TennisMatchSnapshot, match_data: dict) -> TennisFeatureSnapshot:
    market = match_data.get("market") or {}
    player_a = match_data.get("player_a") or {}
    player_b = match_data.get("player_b") or {}
    context = match_data.get("context") or {}
    h2h = match_data.get("h2h") or {}

    return TennisFeatureSnapshot(
        sport="TENNIS",
        player_a=snapshot.player_a,
        player_b=snapshot.player_b,
        tour=snapshot.tour,
        tournament=snapshot.tournament,
        round_name=snapshot.round_name,
        surface=snapshot.surface,
        best_of=snapshot.best_of,
        indoor=snapshot.indoor,
        market_player_a_price=market.get("player_a_price"),
        market_player_b_price=market.get("player_b_price"),
        player_a_rating=float(player_a.get("rating", 1500.0) or 1500.0),
        player_b_rating=float(player_b.get("rating", 1500.0) or 1500.0),
        player_a_surface_rating=float(player_a.get("surface_rating", player_a.get("rating", 1500.0)) or 1500.0),
        player_b_surface_rating=float(player_b.get("surface_rating", player_b.get("rating", 1500.0)) or 1500.0),
        player_a_recent_form=float(player_a.get("recent_form", 0.5) or 0.5),
        player_b_recent_form=float(player_b.get("recent_form", 0.5) or 0.5),
        player_a_hold_pct=float(player_a.get("hold_pct", 0.75) or 0.75),
        player_b_hold_pct=float(player_b.get("hold_pct", 0.75) or 0.75),
        player_a_break_pct=float(player_a.get("break_pct", 0.22) or 0.22),
        player_b_break_pct=float(player_b.get("break_pct", 0.22) or 0.22),
        player_a_rest_days=int(context.get("player_a_rest_days", 1) or 1),
        player_b_rest_days=int(context.get("player_b_rest_days", 1) or 1),
        player_a_last_match_minutes=int(context.get("player_a_last_match_minutes", 0) or 0),
        player_b_last_match_minutes=int(context.get("player_b_last_match_minutes", 0) or 0),
        player_a_travel_zones=int(context.get("player_a_travel_zones", 0) or 0),
        player_b_travel_zones=int(context.get("player_b_travel_zones", 0) or 0),
        player_a_injury_risk=float(player_a.get("injury_risk", 0.0) or 0.0),
        player_b_injury_risk=float(player_b.get("injury_risk", 0.0) or 0.0),
        h2h_player_a_win_pct=h2h.get("player_a_win_pct"),
        h2h_sample=int(h2h.get("sample", 0) or 0),
    )
