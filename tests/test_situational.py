from src.model.situational import (
    NBA_REST_ADV_PER_DAY,
    NBA_SITUATIONAL_WEIGHT,
    SituationalFactors,
    nba_situational,
    nhl_situational,
)


def test_nba_situational_applies_global_weight():
    factors = SituationalFactors(
        home_days_rest=3,
        away_days_rest=1,
        home_is_b2b=False,
        away_is_b2b=False,
    )

    adjustment = nba_situational(factors)
    raw_rest = 2 * NBA_REST_ADV_PER_DAY

    assert adjustment.rest_adj == raw_rest * NBA_SITUATIONAL_WEIGHT
    assert adjustment.total == raw_rest * NBA_SITUATIONAL_WEIGHT
    assert "weight" in adjustment.factors


def test_nhl_situational_remains_undampened():
    factors = SituationalFactors(
        home_days_rest=3,
        away_days_rest=1,
        home_is_b2b=False,
        away_is_b2b=False,
    )

    adjustment = nhl_situational(factors)

    assert adjustment.total > 0
    assert "weight" not in adjustment.factors
