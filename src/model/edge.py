"""
Liquidity-adjusted edge calculation and position sizing.

Formulas from gameplan:
  True_Implied = Ask_Price / (1 - Vig_Estimate)
  Vig_Estimate = (Ask - Bid) / Mid
  Effective_Edge = Your_Probability - True_Implied - Slippage_Cost

Pre-positioning thresholds:
  Same-day: 3% minimum
  24-48h out: 5% minimum
  48h+ out: 7% minimum
"""

from dataclasses import dataclass


@dataclass
class EdgeCalculation:
    """Full edge calculation result."""
    ask_price: float          # market ask price (what you pay)
    bid_price: float          # market bid price
    mid_price: float          # (ask + bid) / 2
    vig_estimate: float       # spread-based vig
    true_implied: float       # vig-adjusted implied probability
    your_probability: float   # model's probability estimate
    slippage_cost: float      # estimated slippage
    effective_edge: float     # final edge after all costs
    position_size: float      # recommended position size (dollar amount)
    position_pct: float       # position as fraction of bankroll
    kelly_full: float         # full Kelly fraction
    kelly_used: float         # fractional Kelly actually applied
    passes_threshold: bool    # whether edge meets pre-positioning minimum
    threshold_used: float     # which threshold was applied
    reasoning: str


def calculate_vig(ask_price: float, bid_price: float) -> float:
    """Calculate vig estimate from bid-ask spread.
    Vig_Estimate = (Ask - Bid) / Mid"""
    mid = (ask_price + bid_price) / 2
    if mid <= 0:
        return 0.0
    return (ask_price - bid_price) / mid


def calculate_true_implied(ask_price: float, vig_estimate: float) -> float:
    """Calculate true implied probability adjusted for vig.
    True_Implied = Ask_Price / (1 - Vig_Estimate)"""
    denominator = 1 - vig_estimate
    if denominator <= 0:
        return ask_price  # fallback: treat ask as implied
    return ask_price / denominator


def calculate_effective_edge(
    your_probability: float,
    true_implied: float,
    slippage_cost: float = 0.005,
) -> float:
    """Calculate effective edge after all costs.
    Effective_Edge = Your_Probability - True_Implied - Slippage_Cost"""
    return your_probability - true_implied - slippage_cost


def get_threshold(hours_to_game: float, sport: str = "NBA") -> float:
    """Get minimum effective edge threshold based on time to game and sport.

    NBA: Aggressive — model Brier 0.176, 78% BET accuracy.
      Same-day: 1.5%  (was 3% — passing too many +EV spots)
      24-48h: 3%
      48h+: 5%

    NHL: Conservative — model Brier 0.240, 60% BET accuracy.
      Same-day: 4%  (raised from 3% — too many false edges)
      24-48h: 6%
      48h+: 8%
    """
    sport = sport.upper() if sport else "NBA"

    if sport == "NHL":
        if hours_to_game <= 12:
            return 0.04
        elif hours_to_game <= 48:
            return 0.06
        else:
            return 0.08
    else:  # NBA
        if hours_to_game <= 12:
            return 0.015
        elif hours_to_game <= 48:
            return 0.03
        else:
            return 0.05


def kelly_fraction(
    model_prob: float,
    ask_price: float,
) -> float:
    """Calculate full Kelly criterion for a Polymarket binary market.

    You pay `ask_price` per share. If you win, the share pays $1.
    Net odds: b = (1 - ask) / ask  (profit per dollar risked)
    Kelly:  f* = (p * b - q) / b = (p - ask) / (1 - ask)

    Returns fraction of bankroll (can be negative if edge is negative).
    """
    if ask_price <= 0 or ask_price >= 1:
        return 0.0
    return (model_prob - ask_price) / (1 - ask_price)


# Position sizing config — all percentage-based, scales with bankroll
KELLY_FRACTION = 0.25        # 1/4 Kelly — conservative
MAX_POSITION_PCT = 0.05      # 5% of bankroll cap (NBA)
MAX_POSITION_PCT_NHL = 0.025 # 2.5% cap for NHL (until model improves)
MAX_POSITION_PCT_A2 = 0.025  # 2.5% cap for conditional (A2) trades
MIN_POSITION_PCT = 0.01      # 1% floor — always bet at least this if edge passes
POLYMARKET_MIN_ORDER = 1.0   # Polymarket absolute minimum order ($1)


def calculate_position_size(
    model_prob: float,
    ask_price: float,
    bankroll: float = 100.0,
    edge_type: str = "B",
    kelly_mult: float = KELLY_FRACTION,
    sport: str = "NBA",
) -> float:
    """Calculate position size using fractional Kelly criterion.

    Pure percentage-based — scales with bankroll AND model confidence.
    Higher confidence = bigger bet. Bigger bankroll = bigger bet.

    Sport-specific caps:
      NBA: 5% cap, 1% floor (proven model, Brier 0.176)
      NHL: 2.5% cap, 1% floor (weak model, Brier 0.240)

    Examples ($100 bankroll):
      NBA 65% model vs 55% ask: Kelly=22%, 1/4=5.6% → cap 5% = $5.00
      NBA 58% model vs 55% ask: Kelly=6.7%, 1/4=1.7% = $1.70
      NHL 60% model vs 52% ask: Kelly=16.7%, 1/4=4.2% → cap 2.5% = $2.50

    Returns dollar amount to bet (not fraction).
    """
    full_kelly = kelly_fraction(model_prob, ask_price)

    if full_kelly <= 0:
        return 0.0

    frac = full_kelly * kelly_mult

    # Apply sport-specific cap
    if edge_type == "A2":
        cap = MAX_POSITION_PCT_A2
    elif sport.upper() == "NHL":
        cap = MAX_POSITION_PCT_NHL
    else:
        cap = MAX_POSITION_PCT
    frac = min(frac, cap)

    # Apply floor
    frac = max(frac, MIN_POSITION_PCT)

    amount = frac * bankroll

    # Only enforce Polymarket's absolute minimum ($1)
    if amount < POLYMARKET_MIN_ORDER:
        amount = POLYMARKET_MIN_ORDER

    return round(amount, 2)


def full_edge_calculation(
    your_probability: float,
    ask_price: float,
    bid_price: float,
    hours_to_game: float = 6.0,
    slippage_cost: float = 0.005,
    bankroll: float = 100.0,
    edge_type: str = "B",
    sport: str = "NBA",
) -> EdgeCalculation:
    """Run the complete edge calculation pipeline.

    Args:
        your_probability: Model's estimated probability (0-1)
        ask_price: Market ask price (0-1)
        bid_price: Market bid price (0-1)
        hours_to_game: Hours until game starts
        slippage_cost: Estimated slippage (default 0.5%)
        bankroll: Current bankroll in USD
        edge_type: Edge classification (A1, A2, B, C)
        sport: "NBA" or "NHL" (sport-specific thresholds)

    Returns:
        EdgeCalculation with full breakdown
    """
    mid_price = (ask_price + bid_price) / 2
    vig = calculate_vig(ask_price, bid_price)
    true_implied = calculate_true_implied(ask_price, vig)
    effective_edge = calculate_effective_edge(your_probability, true_implied, slippage_cost)
    threshold = get_threshold(hours_to_game, sport=sport)
    passes = effective_edge >= threshold

    # Kelly sizing
    full_kelly = kelly_fraction(your_probability, ask_price)
    if passes and full_kelly > 0:
        position_usd = calculate_position_size(
            your_probability, ask_price, bankroll, edge_type, sport=sport,
        )
        position_pct = position_usd / bankroll if bankroll > 0 else 0.0
        cap = MAX_POSITION_PCT_A2 if edge_type == "A2" else MAX_POSITION_PCT
        kelly_used = min(max(full_kelly * KELLY_FRACTION, MIN_POSITION_PCT), cap)
    else:
        position_usd = 0.0
        position_pct = 0.0
        kelly_used = 0.0

    if passes:
        reasoning = (
            f"Edge {effective_edge:.1%} meets {threshold:.0%} threshold "
            f"({hours_to_game:.0f}h to game). "
            f"Kelly: {full_kelly:.1%} full → {kelly_used:.1%} (1/4). "
            f"Bet: ${position_usd:.2f} ({position_pct:.1%} of bankroll)."
        )
    else:
        reasoning = (
            f"Edge {effective_edge:.1%} below {threshold:.0%} threshold "
            f"({hours_to_game:.0f}h to game). SKIP or resize."
        )

    return EdgeCalculation(
        ask_price=ask_price,
        bid_price=bid_price,
        mid_price=mid_price,
        vig_estimate=vig,
        true_implied=true_implied,
        your_probability=your_probability,
        slippage_cost=slippage_cost,
        effective_edge=effective_edge,
        position_size=position_usd,
        position_pct=position_pct,
        kelly_full=full_kelly,
        kelly_used=kelly_used,
        passes_threshold=passes,
        threshold_used=threshold,
        reasoning=reasoning,
    )
