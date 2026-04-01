"""
Layer 3: Information edge.

Scenario-based conditional logic per player tier.
Quantifies the probability impact of injuries/news based on
how important the player is to their team.

All adjustments are from the HOME team's perspective.
"""

from dataclasses import dataclass
from enum import Enum


class PlayerTier(Enum):
    """Player importance tiers with associated probability impact ranges."""
    TIER_1 = 1  # MVP-caliber / franchise player (e.g., McDavid, Jokic)
    TIER_2 = 2  # All-Star level (e.g., Draisaitl, Tatum)
    TIER_3 = 3  # Quality starter (e.g., solid 2nd line center, starting PG)
    TIER_4 = 4  # Role player (e.g., 4th liner, bench guard)


class InjuryStatus(Enum):
    """Player availability status."""
    OUT = "out"                 # confirmed out
    DOUBTFUL = "doubtful"      # likely out (75%+ chance of missing)
    QUESTIONABLE = "questionable"  # uncertain (50/50)
    PROBABLE = "probable"      # likely playing (75%+ chance)
    ACTIVE = "active"          # confirmed playing


@dataclass
class PlayerImpact:
    """A single player's impact on win probability."""
    player_name: str
    team: str  # "home" or "away"
    tier: PlayerTier
    status: InjuryStatus
    adjustment: float  # probability adjustment (from home perspective)
    reasoning: str


# Impact ranges by tier (absolute value, applied based on in/out status)
# These are the probability swings when a player is OUT vs their baseline
TIER_IMPACT = {
    PlayerTier.TIER_1: (0.05, 0.08),   # 5-8% swing
    PlayerTier.TIER_2: (0.03, 0.05),   # 3-5% swing
    PlayerTier.TIER_3: (0.01, 0.03),   # 1-3% swing
    PlayerTier.TIER_4: (0.005, 0.01),  # 0.5-1% swing
}

# How much of the full impact applies based on status
# OUT = full impact, QUESTIONABLE = 50% weighted, etc.
STATUS_WEIGHT = {
    InjuryStatus.OUT: 1.0,
    InjuryStatus.DOUBTFUL: 0.75,
    InjuryStatus.QUESTIONABLE: 0.50,
    InjuryStatus.PROBABLE: 0.15,
    InjuryStatus.ACTIVE: 0.0,
}


def calculate_player_impact(
    player_name: str,
    team: str,
    tier: PlayerTier,
    status: InjuryStatus,
    impact_estimate: float | None = None,
) -> PlayerImpact:
    """Calculate a single player's impact on home win probability.

    Args:
        player_name: Player's name
        team: "home" or "away"
        tier: Player's importance tier
        status: Current injury/availability status
        impact_estimate: Override the default tier impact (0-1 scale).
            If None, uses the midpoint of the tier's range.

    Returns:
        PlayerImpact with the adjustment from home team's perspective.
    """
    if impact_estimate is not None:
        base_impact = impact_estimate
    else:
        low, high = TIER_IMPACT[tier]
        base_impact = (low + high) / 2  # midpoint of range

    weight = STATUS_WEIGHT[status]
    raw_adjustment = base_impact * weight

    # Direction: if home player is out, home probability DECREASES
    # If away player is out, home probability INCREASES
    if team == "home":
        adjustment = -raw_adjustment
    else:
        adjustment = raw_adjustment

    # Build reasoning
    if status == InjuryStatus.ACTIVE:
        reasoning = f"{player_name} ({tier.name}) is active — no adjustment"
    else:
        direction = "reduces" if team == "home" else "increases"
        reasoning = (
            f"{player_name} ({tier.name}, {status.value}) "
            f"{direction} home win prob by {abs(adjustment):.1%}"
        )

    return PlayerImpact(
        player_name=player_name,
        team=team,
        tier=tier,
        status=status,
        adjustment=adjustment,
        reasoning=reasoning,
    )


@dataclass
class InformationEdge:
    """Combined information edge from all player impacts."""
    total: float  # net adjustment to home win probability
    player_impacts: list[PlayerImpact]
    reasoning: str


def calculate_information_edge(impacts: list[PlayerImpact]) -> InformationEdge:
    """Combine multiple player impacts into a single information edge.

    Applies diminishing returns for multiple injuries at the same tier:
    1st at tier = 100%, 2nd = 50%, 3rd+ = 25%.

    Args:
        impacts: List of PlayerImpact objects

    Returns:
        InformationEdge with total adjustment and reasoning
    """
    # Count how many injuries we've seen per (team, tier) for diminishing returns
    tier_counts: dict[tuple[str, PlayerTier], int] = {}
    total = 0.0
    for p in impacts:
        if p.status == InjuryStatus.ACTIVE:
            continue
        key = (p.team, p.tier)
        count = tier_counts.get(key, 0)
        if count == 0:
            scale = 1.0
        elif count == 1:
            scale = 0.5
        else:
            scale = 0.25
        tier_counts[key] = count + 1
        total += p.adjustment * scale

    # Add back active players (0 adjustment anyway)
    # total already accounts for all non-active impacts with diminishing returns

    # Cap total information edge at ±15% to prevent model from being
    # dominated by injury stacking
    total = max(-0.15, min(0.15, total))

    active_impacts = [p for p in impacts if p.status != InjuryStatus.ACTIVE]
    if active_impacts:
        summary_parts = [p.reasoning for p in active_impacts]
        reasoning = "; ".join(summary_parts)
    else:
        reasoning = "No material injury/news impacts"

    return InformationEdge(
        total=total,
        player_impacts=impacts,
        reasoning=reasoning,
    )
