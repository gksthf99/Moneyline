from __future__ import annotations

from dataclasses import dataclass, field
import math

from src.features.tennis_features import TennisFeatureSnapshot


SURFACE_WEIGHT = 0.60
FORM_WEIGHT = 35.0
SERVE_RETURN_WEIGHT = 120.0
REST_DAY_WEIGHT = 12.0
MATCH_LOAD_WEIGHT = 0.08
TRAVEL_ZONE_WEIGHT = 10.0
INJURY_RISK_WEIGHT = 90.0
H2H_WEIGHT = 25.0
BEST_OF_FIVE_BONUS = 18.0
INDOOR_BONUS = 6.0
RATING_SCALE = 180.0


@dataclass(frozen=True)
class TennisProbabilityDecomposition:
    player_a: str
    player_b: str
    tour: str
    base_probability: float
    surface_adjustment: float
    fatigue_adjustment: float
    information_adjustment: float
    final_probability: float
    reasoning: tuple[str, ...] = field(default_factory=tuple)


def _logistic(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def _clamp(probability: float) -> float:
    return max(0.05, min(0.95, probability))


def predict_tennis_match(feature: TennisFeatureSnapshot) -> TennisProbabilityDecomposition:
    rating_spread = feature.player_a_rating - feature.player_b_rating
    surface_spread = feature.player_a_surface_rating - feature.player_b_surface_rating
    form_spread = feature.player_a_recent_form - feature.player_b_recent_form
    serve_return_spread = (
        (feature.player_a_hold_pct - feature.player_b_hold_pct)
        + (feature.player_a_break_pct - feature.player_b_break_pct)
    )

    rating_only_probability = _clamp(_logistic(rating_spread / RATING_SCALE))
    base_strength = rating_spread + (surface_spread * SURFACE_WEIGHT)
    base_probability = _clamp(_logistic(base_strength / RATING_SCALE))

    fatigue_score = (
        (feature.player_a_rest_days - feature.player_b_rest_days) * REST_DAY_WEIGHT
        - (feature.player_a_last_match_minutes - feature.player_b_last_match_minutes) * MATCH_LOAD_WEIGHT
        - (feature.player_a_travel_zones - feature.player_b_travel_zones) * TRAVEL_ZONE_WEIGHT
    )
    information_score = (
        form_spread * FORM_WEIGHT
        + serve_return_spread * SERVE_RETURN_WEIGHT
        - (feature.player_a_injury_risk - feature.player_b_injury_risk) * INJURY_RISK_WEIGHT
    )

    if feature.h2h_player_a_win_pct is not None and feature.h2h_sample >= 2:
        information_score += (feature.h2h_player_a_win_pct - 0.5) * H2H_WEIGHT

    if feature.best_of == 5:
        information_score += BEST_OF_FIVE_BONUS * (rating_spread / 100.0)
    if feature.indoor:
        information_score += INDOOR_BONUS * serve_return_spread

    fatigued_probability = _clamp(_logistic((base_strength + fatigue_score) / RATING_SCALE))
    final_strength = base_strength + fatigue_score + information_score
    final_probability = _clamp(_logistic(final_strength / RATING_SCALE))

    reasoning = (
        f"rating_spread={rating_spread:+.1f}",
        f"surface_spread={surface_spread:+.1f}",
        f"fatigue_score={fatigue_score:+.1f}",
        f"information_score={information_score:+.1f}",
    )

    return TennisProbabilityDecomposition(
        player_a=feature.player_a,
        player_b=feature.player_b,
        tour=feature.tour,
        base_probability=base_probability,
        surface_adjustment=base_probability - rating_only_probability,
        fatigue_adjustment=fatigued_probability - base_probability,
        information_adjustment=final_probability - fatigued_probability,
        final_probability=final_probability,
        reasoning=reasoning,
    )
