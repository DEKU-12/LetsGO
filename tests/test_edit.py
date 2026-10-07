"""Editing a finished plan by chatting, on the mock backend."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from backend.api import app
from backend.edit import NONE, edit_trip, keep_untouched_days, route_edit
from backend.graph import plan_trip
from backend.llm import LLM
from backend.state import ActivityBlock, ItineraryDay, ItineraryOutput

TRIP = "5 days in Japan, mid-range budget, love food and history, need hotels and flights"


@pytest.fixture(scope="module")
def llm() -> LLM:
    return LLM(provider="mock")


@pytest.fixture(scope="module")
def planned(llm: LLM):
    return plan_trip(TRIP, llm)


def _days(*titles: str) -> ItineraryOutput:
    return ItineraryOutput(days=[
        ItineraryDay(day=i, blocks=[ActivityBlock(time="09:00", title=t)])
        for i, t in enumerate(titles, start=1)
    ])


def test_only_the_named_days_are_taken_from_the_edit() -> None:
    old = _days("Temple", "Market", "Museum")
    new = _days("CHANGED", "CHANGED", "Quiet garden")

    merged = keep_untouched_days(old, new, [3])

    assert [d.blocks[0].title for d in merged.days] == ["Temple", "Market", "Quiet garden"]


def test_a_whole_trip_edit_takes_every_day() -> None:
    new = _days("A", "B")
    assert keep_untouched_days(_days("x", "y"), new, []) is new


@pytest.mark.parametrize(("message", "agent", "days"), [
    ("less walking on day 3", "itinerary", [3]),
    ("swap days 1 and 2", "itinerary", [1, 2]),
    ("a cheaper hotel please", "accommodation", []),
    ("no taxis, public transport only", "transport", []),
    ("actually go to Korea instead", NONE, []),
])
def test_edits_are_routed_to_the_right_section(planned, llm, message, agent, days) -> None:
    decision = route_edit(planned, message, llm)
    assert (decision["agent"], decision["days"]) == (agent, days)


def test_out_of_range_days_are_dropped(planned, llm) -> None:
    assert route_edit(planned, "change day 9", llm)["days"] == []


def test_an_itinerary_edit_reruns_only_what_it_needs(planned, llm) -> None:
    edited, reply = edit_trip(planned, "less walking on day 3", llm)

    assert reply == ""
    assert edited["trace"][0] == "edit_router"
    assert "itinerary" in edited["trace"] and "check_itinerary" in edited["trace"]
    assert "destination_research" not in edited["trace"]
    assert "accommodation" not in edited["trace"]
    assert edited["final_plan"]
    # Untouched days are the originals, object for object.
    for before, after in zip(planned["itinerary"].days, edited["itinerary"].days):
        if before.day != 3:
            assert after == before


def test_editing_does_not_change_the_original(planned, llm) -> None:
    trace_before = list(planned["trace"])
    edit_trip(planned, "a cheaper hotel please", llm)
    assert planned["trace"] == trace_before
    assert planned.get("edit_request") is None


def test_a_hotel_edit_passes_the_request_to_the_agent(planned, llm, monkeypatch) -> None:
    prompts: list[str] = []
    original = LLM.json

    def spy(self: LLM, **kwargs: Any) -> Any:
        prompts.append(kwargs["prompt"])
        return original(self, **kwargs)

    monkeypatch.setattr(LLM, "json", spy)
    edited, _ = edit_trip(planned, "a cheaper hotel please", llm)

    assert edited["trace"][:2] == ["edit_router", "accommodation"]
    assert any("asked for this change: a cheaper hotel please" in p for p in prompts)


def test_a_new_destination_is_declined_not_applied(planned, llm) -> None:
    edited, reply = edit_trip(planned, "actually go to Korea instead", llm)
    assert edited is None
    assert reply


# -- API ----------------------------------------------------------------------


@pytest.fixture
def client() -> TestClient:
    with TestClient(app) as c:
        yield c


OWNER = "33333333-aaaa-4bbb-8ccc-000000000003"


def _plan(client: TestClient) -> dict[str, Any]:
    return client.post(
        "/api/plan", json={"request": TRIP, "provider": "mock", "user_id": OWNER}
    ).json()


def test_an_edit_is_saved_as_a_new_version(client: TestClient) -> None:
    original = _plan(client)

    body = client.post(
        f"/api/trips/{original['trip_id']}/edit",
        json={"message": "less walking on day 3", "provider": "mock", "user_id": OWNER},
    ).json()

    assert body["type"] == "done"
    assert body["trip_id"] != original["trip_id"]
    assert body["parent_id"] == original["trip_id"]
    assert (body["edited"]["section"], body["edited"]["days"]) == ("itinerary", [3])

    saved = client.get(f"/api/trips/{body['trip_id']}?user_id={OWNER}").json()
    assert saved["parent_id"] == original["trip_id"]
    assert saved["edit_message"] == "less walking on day 3"
    # The original is still there, unchanged: that is the undo.
    first = client.get(f"/api/trips/{original['trip_id']}?user_id={OWNER}").json()
    assert first["plan"] == original["plan"]


def test_a_declined_edit_saves_nothing(client: TestClient) -> None:
    original = _plan(client)
    newest = client.get(f"/api/trips?user_id={OWNER}&limit=1").json()[0]["id"]

    body = client.post(
        f"/api/trips/{original['trip_id']}/edit",
        json={"message": "go to Korea instead", "provider": "mock", "user_id": OWNER},
    ).json()

    assert body["type"] == "reply" and body["reply"]
    assert client.get(f"/api/trips?user_id={OWNER}&limit=1").json()[0]["id"] == newest


def test_editing_an_unknown_trip_is_a_404(client: TestClient) -> None:
    response = client.post("/api/trips/999999999/edit",
                           json={"message": "x", "provider": "mock", "user_id": OWNER})
    assert response.status_code == 404


def test_a_model_failure_mid_edit_changes_nothing(planned, llm, monkeypatch) -> None:
    from backend.llm import LLMError

    original = LLM.json

    def failing(self: LLM, **kwargs: Any) -> Any:
        if kwargs["task"] in ("itinerary", "accommodation"):
            raise LLMError(f"{self.provider}: RateLimitError: slow down")
        return original(self, **kwargs)

    monkeypatch.setattr(LLM, "json", failing)

    for message in ("less walking on day 3", "a cheaper hotel please"):
        edited, reply = edit_trip(planned, message, llm)
        assert edited is None
        assert "Nothing was changed" in reply


def test_naming_every_day_means_the_whole_trip(planned, llm) -> None:
    every_day = " ".join(f"day {d}" for d in range(1, len(planned["itinerary"].days) + 1))
    assert route_edit(planned, f"start later on {every_day}", llm)["days"] == []


# -- trip-day help ------------------------------------------------------------

RAIN = {"description": "moderate rain", "temp_c": 14.0, "bad": True}
SUN = {"description": "clear sky", "temp_c": 22.0, "bad": False}


def test_today_in_a_message_means_the_day_the_traveller_is_on(planned, llm) -> None:
    decision = route_edit(planned, "it's raining, rearrange today", llm, today=2)
    assert (decision["agent"], decision["days"]) == ("itinerary", [2])


def test_trip_day_comes_from_the_start_date(planned) -> None:
    from datetime import date

    from backend.edit import trip_day

    dated = {**planned, "params": planned["params"].model_copy(update={"start_date": "2026-10-01"})}
    assert trip_day(dated, date(2026, 10, 3)) == 3
    assert trip_day(dated, date(2026, 9, 30)) is None  # before the trip
    assert trip_day(dated, date(2026, 12, 1)) is None  # after it
    assert trip_day(planned, date(2026, 10, 3)) is None  # no start date


def test_fine_weather_changes_nothing(planned, llm) -> None:
    from backend.edit import check_today

    edited, reply = check_today(planned, 2, SUN, llm)
    assert edited is None and "clear sky" in reply


def test_unknown_weather_is_said_not_guessed(planned, llm) -> None:
    from backend.edit import check_today

    unknown = {"description": None, "temp_c": None, "bad": False}
    edited, reply = check_today(planned, 2, unknown, llm)
    assert edited is None and "isn't available" in reply


def test_rain_rearranges_only_today_with_the_facts(planned, llm, monkeypatch) -> None:
    from backend.edit import check_today

    prompts: list[str] = []
    original = LLM.json

    def spy(self: LLM, **kwargs: Any) -> Any:
        prompts.append(kwargs["prompt"])
        return original(self, **kwargs)

    monkeypatch.setattr(LLM, "json", spy)
    edited, reply = check_today(planned, 2, RAIN, llm)

    assert reply == ""
    assert edited["meta"]["edit"]["days"] == [2]
    assert any("moderate rain, 14°C" in p for p in prompts)
    for before, after in zip(planned["itinerary"].days, edited["itinerary"].days):
        if before.day != 2:
            assert after == before


def test_today_endpoint_needs_to_know_the_day(client: TestClient) -> None:
    trip_id = _plan(client)["trip_id"]
    response = client.post(f"/api/trips/{trip_id}/today",
                           json={"provider": "mock", "user_id": OWNER})
    assert response.status_code == 400
    assert "Which day" in response.json()["detail"]


def test_today_endpoint_rearranges_in_the_rain(client: TestClient, monkeypatch) -> None:
    from backend.adapters.base import AdapterResult
    from backend.adapters.weather import WeatherAdapter

    trip_id = _plan(client)["trip_id"]
    monkeypatch.setattr(
        WeatherAdapter, "fetch", lambda self, **_: AdapterResult(RAIN, "live", "openweather")
    )

    body = client.post(
        f"/api/trips/{trip_id}/today", json={"today": 2, "provider": "mock", "user_id": OWNER}
    ).json()

    assert body["type"] == "done"
    assert body["parent_id"] == trip_id
    assert body["edited"]["days"] == [2]
    assert body["weather"]["description"] == "moderate rain"


def test_an_edit_that_changes_nothing_says_so(planned, llm) -> None:
    from backend.edit import check_today

    # The mock itinerary ignores instructions, so the day comes back as it was.
    edited, _ = check_today(planned, 2, RAIN, llm)
    assert edited["meta"]["edit"]["changed"] is False
