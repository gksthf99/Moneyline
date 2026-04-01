from scripts.export_tennis_historical_dataset import _row_from_match


def test_row_from_match_contains_backtest_fields():
    match = {
        "id": "m1",
        "tour": "ATP",
        "player_a_id": "a1",
        "player_b_id": "b1",
        "player_a": "Player A",
        "player_b": "Player B",
        "tournament": "Miami Open",
        "round_name": "Final",
        "surface": "hard",
        "best_of": 3,
        "scheduled_time": "2026-03-29T18:00:00Z",
        "indoor": False,
        "status": "closed",
    }
    match_data = {
        "player_a": {"rating": 1800, "surface_rating": 1820, "recent_form": 0.7, "hold_pct": 0.83, "break_pct": 0.24, "injury_risk": 0.01},
        "player_b": {"rating": 1750, "surface_rating": 1740, "recent_form": 0.6, "hold_pct": 0.79, "break_pct": 0.20, "injury_risk": 0.05},
        "context": {"player_a_rest_days": 2, "player_b_rest_days": 1, "player_a_last_match_minutes": 90, "player_b_last_match_minutes": 140, "player_a_travel_zones": 0, "player_b_travel_zones": 1},
        "h2h": {"player_a_win_pct": 0.67, "sample": 3},
        "market": {"player_a_price": 0.58, "player_b_price": 0.45},
    }

    row = _row_from_match(match, match_data, True)

    assert row["tour"] == "ATP"
    assert row["player_a_rating"] == 1800
    assert row["market_player_a_price"] == 0.58
    assert row["player_a_won"] is True
