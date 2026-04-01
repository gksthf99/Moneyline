from __future__ import annotations

from src.agents import research_agent


def _sample_match() -> dict:
    return {
        "id": "m1",
        "tour": "ATP",
        "player_a": "Carlos Alcaraz",
        "player_b": "Jannik Sinner",
        "tournament": "Miami Open",
        "round_name": "Semifinal",
        "surface": "hard",
        "best_of": 3,
        "scheduled_time": "2026-03-29T18:00:00Z",
    }


def _sample_artifacts() -> TennisPredictionArtifacts:
    match = _sample_match()
    match_data = {
        "player_a": {"rating": 1750, "surface_rating": 1770, "recent_form": 0.7},
        "player_b": {"rating": 1725, "surface_rating": 1745, "recent_form": 0.66},
        "context": {},
        "h2h": {},
        "market": {"player_a_price": 0.56, "player_b_price": 0.44},
    }
    from src.services.tennis_prediction import predict_tennis

    return predict_tennis(match, match_data)


def test_post_to_discord_tennis_thread_title(monkeypatch) -> None:
    calls = []

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append((url, json))

        class Resp:
            status_code = 201

            @staticmethod
            def json():
                return {"id": "thread-1"}

            text = ""

        return Resp()

    monkeypatch.setattr(research_agent, "_load_discord_config", lambda: {
        "channels": {"research_tennis": "chan-1"},
        "tokens": {"research": "token"},
    })
    monkeypatch.setattr(research_agent.requests, "post", fake_post)

    thread_id = research_agent.post_to_discord(
        "report",
        "TENNIS",
        game=_sample_match(),
    )

    assert thread_id == "thread-1"
    assert "Carlos Alcaraz vs Jannik Sinner" in calls[0][1]["name"]


def test_run_tennis_session(monkeypatch) -> None:
    monkeypatch.setattr(research_agent, "load_tennis_slate", lambda match_date=None, tour=None: [_sample_match()])
    monkeypatch.setattr(research_agent, "build_tennis_match_data", lambda match: {
        "player_a": {"rating": 1750, "surface_rating": 1770, "recent_form": 0.7},
        "player_b": {"rating": 1725, "surface_rating": 1745, "recent_form": 0.66},
        "context": {},
        "h2h": {},
        "market": {"player_a_price": 0.56, "player_b_price": 0.44},
    })
    monkeypatch.setattr(research_agent, "post_to_discord", lambda report, sport, game=None: "thread-1")
    persisted = []
    monkeypatch.setattr(research_agent, "persist_tennis_prediction", lambda artifacts, repository=None: persisted.append(artifacts))

    results = research_agent.run_tennis_session(tour="ATP")

    assert len(results) == 1
    assert results[0]["sport"] == "TENNIS"
    assert results[0]["tour"] == "ATP"
    assert results[0]["posted"] is True
    assert len(persisted) == 1
