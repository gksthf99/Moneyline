from src.data.tennis_api import (
    _data_auth_headers,
    _normalize_market,
    _normalize_match,
    _normalize_player_profile,
    _normalize_sportradar_summary,
    _odds_auth_headers,
    build_match_data,
)


def test_normalize_match_supports_tour_and_surface():
    match = _normalize_match(
        {
            "match_id": "wta-1",
            "tour": "WTA",
            "player1": "Iga Swiatek",
            "player2": "Aryna Sabalenka",
            "event_name": "Madrid Open",
            "round": "Final",
            "surface": "Clay",
            "best_of": 3,
            "start_time": "2026-05-02T16:00:00Z",
        }
    )

    assert match["tour"] == "WTA"
    assert match["player_a"] == "Iga Swiatek"
    assert match["surface"] == "clay"
    assert match["player_a_id"] is None


def test_normalize_player_profile_sets_safe_defaults():
    profile = _normalize_player_profile({}, player_name="Player A", tour="ATP")

    assert profile["tour"] == "ATP"
    assert profile["rating"] == 1500.0
    assert profile["recent_form"] == 0.5


def test_normalize_market_maps_generic_price_keys():
    market = _normalize_market({"home_price": 0.61, "away_price": 0.43, "id": "m1"})

    assert market["player_a_price"] == 0.61
    assert market["player_b_price"] == 0.43
    assert market["market_id"] == "m1"


def test_normalize_sportradar_summary_extracts_ids_and_tour():
    match = _normalize_sportradar_summary(
        {
            "sport_event": {
                "id": "sr:sport_event:1",
                "start_time": "2026-05-02T16:00:00Z",
                "status": "closed",
                "best_of": 3,
                "competitors": [
                    {"id": "sr:competitor:1", "name": "Iga Swiatek"},
                    {"id": "sr:competitor:2", "name": "Aryna Sabalenka"},
                ],
                "sport_event_context": {
                    "competition": {"name": "Madrid Open", "gender": "WTA"},
                },
                "round": {"name": "Final"},
            },
            "conditions": {"ground": "clay"},
        }
    )

    assert match["id"] == "sr:sport_event:1"
    assert match["player_a_id"] == "sr:competitor:1"
    assert match["player_b_id"] == "sr:competitor:2"
    assert match["tour"] == "WTA"


def test_build_match_data_returns_safe_defaults_without_provider():
    payload = build_match_data(
        {
            "id": "atp-1",
            "tour": "ATP",
            "player_a": "Player A",
            "player_b": "Player B",
            "player_a_rest_days": 2,
            "player_b_rest_days": 1,
        }
    )

    assert payload["player_a"]["tour"] == "ATP"
    assert payload["player_b"]["tour"] == "ATP"
    assert payload["market"] == {}
    assert payload["context"]["player_a_rest_days"] == 2


def test_tennis_auth_headers_are_split(monkeypatch):
    monkeypatch.setattr("src.data.tennis_api.TENNIS_DATA_API_KEY", "data-key")
    monkeypatch.setattr("src.data.tennis_api.TENNIS_ODDS_API_KEY", "odds-key")

    assert _data_auth_headers() == {"Authorization": "Bearer data-key"}
    assert _odds_auth_headers() == {"Authorization": "Bearer odds-key"}
