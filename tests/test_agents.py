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


def test_accommodation_respects_the_budget(llm: LLM) -> None:
    result = accommodation(_state(), llm)
    output = result["accommodation"]

    assert output.options
    assert output.nightly_budget_usd
    assert all(o.price_per_night_usd <= output.nightly_budget_usd * 1.25 for o in output.options)


def test_accommodation_records_that_its_data_is_mock(llm: LLM) -> None:
    assert accommodation(_state(), llm)["sources"] == ["lodging:mock"]


def test_accommodation_drops_options_not_on_the_shortlist(llm: LLM, monkeypatch) -> None:
    """The model must choose from the candidates, not invent a hotel."""
    monkeypatch.setattr(
        LLM, "json",
        lambda self, **kw: {"options": [{"name": "Hotel Imaginary", "why": "made up"}]},
    )
    result = accommodation(_state(), llm)

    assert all(o.name != "Hotel Imaginary" for o in result["accommodation"].options)
    assert any("not on the shortlist" in e for e in result["errors"])


# -- transport --------------------------------------------------------------


def test_transport_returns_both_legs_and_flags_mock_data(llm: LLM) -> None:
    result = transport(_state(), llm)

    assert result["transport"].inbound
    assert result["transport"].local
    assert result["sources"] == ["transport:mock"]


def test_transport_scales_inbound_cost_with_group_size(llm: LLM) -> None:
    params = TripParams(destination="Japan", duration_days=3, travelers=1)
    one = transport(_state(params=params), llm)["transport"].inbound[0].est_cost_usd
    two = transport(_state(), llm)["transport"].inbound[0].est_cost_usd

    assert two == pytest.approx(one * 2)


def test_transport_drops_unknown_modes(llm: LLM, monkeypatch) -> None:
    monkeypatch.setattr(
        LLM, "json",
        lambda self, **kw: {"local": [{"mode": "Teleporter", "description": "instant"}]},
    )
    result = transport(_state(), llm)

    assert all(leg.mode != "Teleporter" for leg in result["transport"].local)
    assert any("not a known mode" in e for e in result["errors"])
