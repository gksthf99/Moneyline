from src.features.game_features import FeatureSnapshot
from src.features.model_features import (
    build_nba_model_features,
    build_nhl_model_features,
)
from src.model.rebuild_spec import get_replacement_model_spec


def test_build_nba_model_features_extracts_expected_spreads():
    feature = FeatureSnapshot(
        sport="NBA",
        home_team="Bulls",
        away_team="Knicks",
        home_win_pct=0.62,
        away_win_pct=0.54,
        home_home_pct=0.68,
        away_away_pct=0.48,
        home_l10_pct=0.7,
        away_l10_pct=0.4,
        home_net_rtg={"season": {"net_rtg": 4.2, "pace": 99.5}, "l10": {"net_rtg": 6.1}},
        away_net_rtg={"season": {"net_rtg": 1.2, "pace": 97.0}, "l10": {"net_rtg": 0.4}},
        rest_differential=1.0,
        home_is_b2b=False,
        away_is_b2b=True,
        away_travel_zones=2,
        market_home_price=0.58,
        player_impact_adj=-0.02,
        injury_count=2,
        clv_residual=0.01,
    )

    row = build_nba_model_features(feature, base_probability=0.57, final_probability=0.56)

    assert row.net_rating_spread == 3.0
    assert row.recent_net_rating_spread == 5.7
    assert row.pace_differential == 2.5
    assert row.player_impact_adjustment == -0.02


def test_build_nhl_model_features_extracts_expected_spreads():
    feature = FeatureSnapshot(
        sport="NHL",
        home_team="Stars",
        away_team="Jets",
        home_win_pct=0.61,
        away_win_pct=0.57,
        home_home_pct=0.65,
        away_away_pct=0.5,
        home_l10_pct=0.6,
        away_l10_pct=0.5,
        home_xgf_pct=0.54,
        away_xgf_pct=0.5,
        home_gf_ga_ratio=0.56,
        away_gf_ga_ratio=0.51,
        rest_differential=1.0,
        home_is_b2b=False,
        away_is_b2b=True,
        away_travel_zones=1,
        market_home_price=0.55,
        goalie_adjustment=0.03,
        player_impact_adj=0.0,
        injury_count=1,
        clv_residual=-0.01,
    )

    row = build_nhl_model_features(feature, base_probability=0.58, final_probability=0.59)

    assert row.xgf_pct_spread == 0.04
    assert row.gf_ga_ratio_spread == 0.05
    assert row.goalie_adjustment == 0.03


def test_replacement_model_spec_is_sport_specific():
    nba = get_replacement_model_spec("NBA")
    nhl = get_replacement_model_spec("NHL")

    assert nba.model_name.startswith("nba_")
    assert nhl.model_name.startswith("nhl_")
    assert "net_rating_spread" in nba.feature_names
    assert "goalie_adjustment" in nhl.feature_names
