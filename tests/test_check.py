"""The schedule check, and saving a trip's full state."""

from __future__ import annotations

from typing import Any

import pytest

from backend.agents.check import MAX_RETRIES, after_check, check_itinerary, find_problems
from backend.db import init_db, load_state, save_trip
from backend.graph import plan_trip
from backend.llm import LLM
from backend.state import (
    ActivityBlock,
    Attraction,
    ItineraryDay,
    ItineraryOutput,
    ResearchOutput,
    TripParams,
    new_state,
)

TOKYO = (35.71, 139.80)
KYOTO = (34.97, 135.77)


def _state(days: list[list[str]], attractions: list[Attraction], wanted: int | None = None):
    state = new_state("test")
    state["params"] = TripParams(destination="Japan", duration_days=wanted or len(days))
    state["research"] = ResearchOutput(destination="Japan", attractions=attractions)
    state["itinerary"] = ItineraryOutput(days=[
        ItineraryDay(day=i, blocks=[ActivityBlock(time="09:00", title=t) for t in titles])
        for i, titles in enumerate(days, start=1)
    ])
    return state


def _place(name: str, hours: float = 2.0, at: tuple[float, float] | None = None) -> Attraction:
    lat, lon = at or (None, None)
    return Attraction(name=name, est_hours=hours, lat=lat, lon=lon)


def test_a_reasonable_schedule_passes() -> None:
    places = [_place("Senso-ji Temple", at=TOKYO), _place("Tokyo National Museum", at=TOKYO)]
    state = _state([["Visit Senso-ji Temple", "Lunch nearby", "Tokyo National Museum"]], places)
    assert find_problems(state) == []


def test_an_overloaded_day_is_flagged() -> None:
    places = [_place("Vatican Museums", 5), _place("Colosseum", 3), _place("Roman Forum", 2)]
    state = _state([["Vatican Museums", "Colosseum", "Roman Forum"]], places)
    assert any("10 hours" in p for p in find_problems(state))


def test_a_repeated_stop_is_flagged() -> None:
    places = [_place("Pantheon", 1)]
    state = _state([["Pantheon"], ["Pantheon again at night"]], places)
    assert any("Pantheon is on day 1 and day 2" in p for p in find_problems(state))


def test_far_apart_stops_on_one_day_are_flagged() -> None:
    places = [_place("Senso-ji Temple", at=TOKYO), _place("Fushimi Inari Shrine", at=KYOTO)]
    state = _state([["Senso-ji Temple", "Fushimi Inari Shrine"]], places)
    assert any("km apart" in p for p in find_problems(state))


def test_missing_and_empty_days_are_flagged() -> None:
    state = _state([["Colosseum"], []], [_place("Colosseum")], wanted=3)
    problems = find_problems(state)
    assert any("Only 2 of 3 days" in p for p in problems)
    assert any("Day 2 is empty" in p for p in problems)


def test_failed_check_sends_the_itinerary_back_until_retries_run_out() -> None:
    places = [_place("Pantheon", 1)]
    state: dict[str, Any] = _state([["Pantheon"], ["Pantheon"]], places)

    for _ in range(MAX_RETRIES):
        state.update(check_itinerary(state))
        assert after_check(state) == "itinerary"

    update = check_itinerary(state)
    state.update(update)
    assert after_check(state) == "supervisor"
    assert any("unresolved" in e for e in update["errors"])
    assert len(state["meta"]["itinerary_checks"]) == MAX_RETRIES + 1


def test_a_full_run_passes_through_the_check() -> None:
    state = plan_trip("5 days in Japan, love food and history", LLM(provider="mock"))
    trace = state["trace"]
    assert trace.index("itinerary") < trace.index("check_itinerary") < trace.index("aggregator")


@pytest.fixture(scope="module", autouse=True)
def _db() -> None:
    init_db()


def test_saved_trip_state_comes_back_with_its_models() -> None:
    state = plan_trip("3 days in Rome, love food", LLM(provider="mock"))
    trip_id = save_trip(state, provider="mock", model="mock", duration_s=0.0)

    loaded = load_state(trip_id)
    assert loaded is not None
    assert loaded["params"] == state["params"]
    assert loaded["itinerary"] == state["itinerary"]
    assert loaded["final_plan"] == state["final_plan"]
    assert isinstance(loaded["research"], ResearchOutput)


def test_loading_an_unknown_trip_returns_none() -> None:
    assert load_state(10**9) is None


def test_one_place_in_two_blocks_counts_once() -> None:
    places = [_place("Nishiki Market", 5, at=KYOTO), _place("Nijo Castle", 3, at=KYOTO)]
    state = _state([["Nishiki Market", "Lunch at Nishiki Market", "Nijo Castle"]], places)
    assert find_problems(state) == []
