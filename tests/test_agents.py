"""The four worker agents, on the mock backend so no key or network is needed."""

from __future__ import annotations

import pytest

from backend.agents.accommodation import accommodation
from backend.agents.itinerary import DEFAULT_DAYS, itinerary
from backend.agents.transport import transport
from backend.llm import LLM
from backend.state import Attraction, ResearchOutput, TripParams

ATTRACTIONS = [
    Attraction(name="Senso-ji Temple", category="landmark", est_hours=1.5),
    Attraction(name="Tsukiji Market", category="food", est_hours=2.0),
    Attraction(name="Tokyo National Museum", category="museum", est_hours=2.5),
    Attraction(name="Nishiki Market", category="food", est_hours=1.5),
]


@pytest.fixture
def llm() -> LLM:
    return LLM(provider="mock")


def _state(**overrides):
    base = {
        "params": TripParams(
            destination="Japan", duration_days=3, budget="mid-range",
            travelers=2, preferences=["food", "history"],
        ),
        "research": ResearchOutput(destination="Japan", attractions=ATTRACTIONS),
    }
    return {**base, **overrides}


# -- itinerary --------------------------------------------------------------


def test_itinerary_matches_the_requested_trip_length(llm: LLM) -> None:
    result = itinerary(_state(), llm)
    assert len(result["itinerary"].days) == 3
    assert [d.day for d in result["itinerary"].days] == [1, 2, 3]


def test_itinerary_falls_back_to_a_default_length(llm: LLM) -> None:
    params = TripParams(destination="Japan", duration_days=None)
    result = itinerary(_state(params=params), llm)
    assert len(result["itinerary"].days) == DEFAULT_DAYS


def test_itinerary_without_research_reports_rather_than_inventing(llm: LLM) -> None:
    result = itinerary(_state(research=ResearchOutput(destination="Japan")), llm)
    assert result["itinerary"].days == []
    assert any("no attractions" in e for e in result["errors"])


def test_itinerary_blocks_stay_in_time_order(llm: LLM) -> None:
    from backend.agents.itinerary import _SLOTS

    for day in itinerary(_state(), llm)["itinerary"].days:
        times = [_SLOTS.index(b.time) for b in day.blocks if b.time in _SLOTS]
        assert times == sorted(times)


# -- accommodation ----------------------------------------------------------


def test_accommodation_recommends_areas_not_listings(llm: LLM) -> None:
    output = accommodation(_state(), llm)["accommodation"]

    assert output.areas
    assert all(a.name and a.why for a in output.areas)
    # No fake inventory: the model has nowhere to put a hotel name or a price.
    assert not hasattr(output, "options")


def test_accommodation_passes_an_edit_request_to_the_model(llm: LLM, monkeypatch) -> None:
    prompts: list[str] = []
    monkeypatch.setattr(
        LLM, "json", lambda self, **kw: prompts.append(kw["prompt"]) or {"areas": []}
    )
    accommodation(_state(edit_request="somewhere quieter"), llm)
    assert "asked for this change: somewhere quieter" in prompts[0]


def test_accommodation_skips_malformed_areas(llm: LLM, monkeypatch) -> None:
    monkeypatch.setattr(
        LLM, "json",
        lambda self, **kw: {"areas": [{"name": "Asakusa", "why": "ok"}, {"why": "no name"}]},
    )
    output = accommodation(_state(), llm)["accommodation"]
    assert [a.name for a in output.areas] == ["Asakusa"]


# -- transport --------------------------------------------------------------


def test_transport_gives_arrival_and_local_advice(llm: LLM) -> None:
    output = transport(_state(), llm)["transport"]

    assert output.arrival
    assert output.local
    assert not hasattr(output, "inbound")  # no fares, real or invented


def test_transport_skips_malformed_modes(llm: LLM, monkeypatch) -> None:
    monkeypatch.setattr(
        LLM, "json",
        lambda self, **kw: {"local": [{"mode": "Metro", "description": "fast"}, {"mode": "x"}]},
    )
    output = transport(_state(), llm)["transport"]
    assert [t.mode for t in output.local] == ["Metro"]
