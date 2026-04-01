"""
Probability decomposition: combines all three layers into a final probability.

Layer 1: Base probability (Elo/SRS)
Layer 2: Situational adjustment (rest, B2B, travel)
Layer 3: Information edge (injuries/news)

Final_Probability = clamp(Base + Situational + Information, 0.05, 0.95)

All three layers stored separately per trade/research record in Supabase.
"""

import logging
from dataclasses import dataclass, asdict

from src.model.baseline import nba_base_probability, nhl_base_probability
from src.model.situational import (
    SituationalFactors, SituationalAdjustment,
    nba_situational, nhl_situational,
)
from src.model.information import InformationEdge, PlayerImpact
from src.model.edge import EdgeCalculation, full_edge_calculation

logger = logging.getLogger(__name__)

# Feature flag: when True, calculate edge for both home and away sides
# and pick whichever passes threshold with the larger edge.
# Default False — current behaviour unchanged until explicitly enabled.
TWO_SIDED_EDGE = True

@dataclass
class ProbabilityDecomposition:
    """Full probability decomposition with all three layers."""
    sport: str
    home_team: str
    away_team: str

    # Layer 1
    base_probability: float

    # Layer 2
    situational_adjustment: SituationalAdjustment

    # Layer 3
    information_edge: InformationEdge

    # Combined
    final_probability: float

    # Edge calculation (if market data available)
    edge: EdgeCalculation | None = None

    # Two-sided edge fields (populated when TWO_SIDED_EDGE is True)
    away_edge: EdgeCalculation | None = None
    bet_side: str | None = None  # "home", "away", or None

    def to_supabase_dict(self) -> dict:
        """Convert to dict suitable for Supabase JSONB storage."""
        d = {
            "base": round(self.base_probability, 4),
            "situational": round(self.situational_adjustment.total, 4),
            "situational_breakdown": {
                "rest": round(self.situational_adjustment.rest_adj, 4),
                "b2b": round(self.situational_adjustment.b2b_adj, 4),
                "travel": round(self.situational_adjustment.travel_adj, 4),
                "form": round(self.situational_adjustment.form_adj, 4),
                "h2h": round(self.situational_adjustment.h2h_adj, 4),
                "goalie": round(self.situational_adjustment.goalie_adj, 4),
            },
            "situational_factors": self.situational_adjustment.factors,
            "information": round(self.information_edge.total, 4),
            "information_reasoning": self.information_edge.reasoning,
            "final": round(self.final_probability, 4),
        }
        if self.bet_side is not None:
            d["bet_side"] = self.bet_side
        return d

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"{self.away_team} @ {self.home_team} ({self.sport})",
            f"  Base (Layer 1):        {self.base_probability:.1%}",
            f"  Situational (Layer 2): {self.situational_adjustment.total:+.1%}",
        ]
        for key, desc in self.situational_adjustment.factors.items():
            lines.append(f"    {key}: {desc}")
        lines.append(f"  Information (Layer 3): {self.information_edge.total:+.1%}")
        if self.information_edge.reasoning != "No material injury/news impacts":
            lines.append(f"    {self.information_edge.reasoning}")
        lines.append(f"  Final probability:     {self.final_probability:.1%} (home win)")
        if self.edge:
            side_label = f" [{self.bet_side}]" if self.bet_side else ""
            lines.append(f"  Home edge:{side_label}")
            lines.append(f"    Market ask:          {self.edge.ask_price:.1%}")
            lines.append(f"    True implied:        {self.edge.true_implied:.1%}")
            lines.append(f"    Effective edge:      {self.edge.effective_edge:+.1%}")
            lines.append(f"    {self.edge.reasoning}")
        if self.away_edge:
            lines.append(f"  Away edge:")
            lines.append(f"    Market ask:          {self.away_edge.ask_price:.1%}")
            lines.append(f"    True implied:        {self.away_edge.true_implied:.1%}")
            lines.append(f"    Effective edge:      {self.away_edge.effective_edge:+.1%}")
            lines.append(f"    {self.away_edge.reasoning}")
        if self.bet_side:
            lines.append(f"  Bet side:              {self.bet_side}")
        return "\n".join(lines)


def _resolve_two_sided_edge(
    final: float,
    home_ask: float | None,
    home_bid: float | None,
    away_ask: float | None,
    away_bid: float | None,
    hours_to_game: float,
    edge_type: str,
    sport: str,
) -> tuple[EdgeCalculation | None, EdgeCalculation | None, str | None]:
    """Calculate edge for both sides, bet the model's predicted winner.

    Returns (home_edge, away_edge, bet_side).
    bet_side always follows the model (final >= 0.5 → home, else → away).
    Edge calc tells us if the bet is worth taking, but never flips direction.
    """
    home_edge = None
    away_edge = None
    bet_side = None

    # Home edge (existing logic)
    if home_ask is not None and home_bid is not None:
        home_edge = full_edge_calculation(
            your_probability=final,
            ask_price=home_ask,
            bid_price=home_bid,
            hours_to_game=hours_to_game,
            edge_type=edge_type,
            sport=sport,
        )

    # Two-sided: calculate away edge, then bet the model's predicted winner
    if TWO_SIDED_EDGE and away_ask is not None and away_bid is not None:
        away_edge = full_edge_calculation(
            your_probability=1 - final,
            ask_price=away_ask,
            bid_price=away_bid,
            hours_to_game=hours_to_game,
            edge_type=edge_type,
            sport=sport,
        )

        # Always bet the side the model predicts to win.
        # Edge calc tells us if that bet is worth taking at market price,
        # but never flips the pick direction.
        bet_side = "home" if final >= 0.5 else "away"

        logger.info(
            "Two-sided edge [%s]: home=%.2f%% (model %.0f%%) | away=%.2f%% (model %.0f%%) -> %s",
            sport,
            (home_edge.effective_edge * 100) if home_edge else 0,
            final * 100,
            away_edge.effective_edge * 100,
            (1 - final) * 100,
            bet_side,
        )
    elif home_edge is not None:
        # Single-sided fallback (no away market data)
        bet_side = "home" if final >= 0.5 else "away"

    return home_edge, away_edge, bet_side


def decompose_nba(
    home_team: str,
    away_team: str,
    home_win_pct: float,
    away_win_pct: float,
    situational: SituationalFactors,
    player_impacts: list[PlayerImpact] | None = None,
    ask_price: float | None = None,
    bid_price: float | None = None,
    hours_to_game: float = 6.0,
    edge_type: str = "B",
    home_home_pct: float | None = None,
    away_away_pct: float | None = None,
    home_net_rtg: dict | None = None,
    away_net_rtg: dict | None = None,
    away_ask_price: float | None = None,
    away_bid_price: float | None = None,
) -> ProbabilityDecomposition:
    """Full NBA probability decomposition."""
    from src.model.information import calculate_information_edge

    # Layer 1 — net rating model (preferred) or win%/Elo (fallback)
    base = nba_base_probability(home_win_pct, away_win_pct,
                                 home_home_pct=home_home_pct,
                                 away_away_pct=away_away_pct,
                                 home_net_rtg=home_net_rtg,
                                 away_net_rtg=away_net_rtg)

    # Layer 2
    sit_adj = nba_situational(situational)

    # Layer 3
    info_edge = calculate_information_edge(player_impacts or [])

    # Combine layers
    raw = base + sit_adj.total + info_edge.total
    final = max(0.05, min(0.95, raw))

    # Edge calculation (home-only or two-sided)
    home_edge, away_edge, bet_side = _resolve_two_sided_edge(
        final=final,
        home_ask=ask_price,
        home_bid=bid_price,
        away_ask=away_ask_price,
        away_bid=away_bid_price,
        hours_to_game=hours_to_game,
        edge_type=edge_type,
        sport="NBA",
    )

    return ProbabilityDecomposition(
        sport="NBA",
        home_team=home_team,
        away_team=away_team,
        base_probability=base,
        situational_adjustment=sit_adj,
        information_edge=info_edge,
        final_probability=final,
        edge=home_edge,
        away_edge=away_edge,
        bet_side=bet_side,
    )


def decompose_nhl(
    home_team: str,
    away_team: str,
    home_win_pct: float,
    away_win_pct: float,
    situational: SituationalFactors,
    player_impacts: list[PlayerImpact] | None = None,
    ask_price: float | None = None,
    bid_price: float | None = None,
    hours_to_game: float = 6.0,
    edge_type: str = "B",
    home_home_pct: float | None = None,
    away_away_pct: float | None = None,
    home_xgf_pct: float | None = None,
    away_xgf_pct: float | None = None,
    home_gf_ga_ratio: float | None = None,
    away_gf_ga_ratio: float | None = None,
    away_ask_price: float | None = None,
    away_bid_price: float | None = None,
) -> ProbabilityDecomposition:
    """Full NHL probability decomposition."""
    from src.model.information import calculate_information_edge

    # Layer 1 — use home/away splits + xGF% + GF/GA if available
    base_result = nhl_base_probability(home_win_pct, away_win_pct,
                                 home_home_pct=home_home_pct,
                                 away_away_pct=away_away_pct,
                                 home_xgf_pct=home_xgf_pct,
                                 away_xgf_pct=away_xgf_pct,
                                 home_gf_ga_ratio=home_gf_ga_ratio,
                                 away_gf_ga_ratio=away_gf_ga_ratio)
    base = base_result["win_prob"]

    # Layer 2
    sit_adj = nhl_situational(situational)

    # Layer 3
    info_edge = calculate_information_edge(player_impacts or [])

    # Combine layers
    raw = base + sit_adj.total + info_edge.total
    final = max(0.05, min(0.95, raw))

    # Edge calculation (home-only or two-sided)
    home_edge, away_edge, bet_side = _resolve_two_sided_edge(
        final=final,
        home_ask=ask_price,
        home_bid=bid_price,
        away_ask=away_ask_price,
        away_bid=away_bid_price,
        hours_to_game=hours_to_game,
        edge_type=edge_type,
        sport="NHL",
    )

    return ProbabilityDecomposition(
        sport="NHL",
        home_team=home_team,
        away_team=away_team,
        base_probability=base,
        situational_adjustment=sit_adj,
        information_edge=info_edge,
        final_probability=final,
        edge=home_edge,
        away_edge=away_edge,
        bet_side=bet_side,
    )
