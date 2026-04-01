from src.features.game_features import build_feature_snapshot


def test_build_feature_snapshot_for_nba():
    game = {"sport": "NBA", "home_team": "Bulls", "away_team": "Knicks"}
    game_data = {
        "sport": "NBA",
        "home_win_pct": 0.61,
        "away_win_pct": 0.53,
        "home_home_record": "20-10",
        "away_away_record": "15-15",
        "home_form": {"record": "7-3"},
        "away_form": {"record": "4-6"},
        "home_net_rtg": {"season": {"net_rtg": 4.2}},
        "away_net_rtg": {"season": {"net_rtg": 1.1}},
        "market": {"home_price": 0.58, "away_price": 0.42, "liquidity": 1250},
        "injuries": [{"player": "A"}, {"player": "B"}],
        "disagreements": ["espn mismatch"],
        "_situational_inputs": {
            "home_days_rest": 2,
            "away_days_rest": 1,
            "home_is_b2b": False,
            "away_is_b2b": True,
            "away_travel_zones": 2,
        },
        "_clv_residual": 0.012,
        "goalie_adjustment": {"value": 0.0},
    }

    class _Decomp:
        class _Info:
            total = -0.02
        information_edge = _Info()

    snapshot = build_feature_snapshot(game, game_data, _Decomp())

    assert snapshot.sport == "NBA"
    assert snapshot.home_home_pct == 20 / 30
    assert snapshot.away_away_pct == 15 / 30
    assert snapshot.home_l10_pct == 0.7
    assert snapshot.market_liquidity == 1250
    assert snapshot.rest_differential == 1
    assert snapshot.away_is_b2b is True
    assert snapshot.clv_residual == 0.012
    assert snapshot.player_impact_adj == -0.02
    assert snapshot.injury_count == 2
    assert snapshot.freshness_disagreements == ("espn mismatch",)


def test_build_feature_snapshot_for_nhl():
    game = {"sport": "NHL", "home_team": "Stars", "away_team": "Avalanche"}
    game_data = {
        "sport": "NHL",
        "home_home_record": "20-8-2",
        "away_away_record": "18-10-2",
        "home_l10": "7-2-1",
        "away_l10": "5-4-1",
        "home_advanced": {"xGF%": "53.4"},
        "away_advanced": {"xGF%": "51.2"},
        "home_gf": 180,
        "home_ga": 150,
        "away_gf": 170,
        "away_ga": 160,
    }

    snapshot = build_feature_snapshot(game, game_data)

    assert snapshot.sport == "NHL"
    assert round(snapshot.home_xgf_pct, 3) == 0.534
    assert round(snapshot.away_xgf_pct, 3) == 0.512
    assert round(snapshot.home_gf_ga_ratio, 3) == round(180 / 330, 3)
    assert round(snapshot.away_gf_ga_ratio, 3) == round(170 / 330, 3)
