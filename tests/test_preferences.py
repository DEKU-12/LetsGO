"""Remembering lasting preferences, on the mock backend."""

from __future__ import annotations

import uuid
from typing import Any

import pytest
from fastapi.testclient import TestClient

from backend.api import app
from backend.graph import plan_trip
from backend.llm import LLM
from backend.preferences import MAX_PREFERENCES, clean, suggest_preferences


@pytest.fixture
def llm() -> LLM:
    return LLM(provider="mock")


@pytest.fixture
def client() -> TestClient:
    with TestClient(app) as c:
        yield c


@pytest.fixture
def user() -> str:
    return str(uuid.uuid4())


def test_lasting_preferences_are_found_and_trip_details_are_not(llm: LLM) -> None:
    found = suggest_preferences(
        "I'm vegetarian. 5 days in Rome for my mum's birthday", llm, known=[]
    )
    assert found == ["vegetarian"]


def test_already_saved_preferences_are_not_suggested_again(llm: LLM) -> None:
    assert suggest_preferences("vegetarian, 3 days in Rome", llm, known=["Vegetarian"]) == []


def test_clean_trims_dedupes_and_caps() -> None:
    assert clean(["  Vegan ", "vegan", "", "kids"]) == ["vegan", "kids"]
    assert len(clean([f"p{i}" for i in range(50)])) == MAX_PREFERENCES


def test_saved_preferences_reach_every_planning_agent(llm: LLM, monkeypatch) -> None:
    prompts: dict[str, str] = {}
    original_json, original_tools = LLM.json, LLM.run_tools

    def spy_json(self: LLM, **kw: Any) -> Any:
        prompts[kw["task"]] = kw["prompt"]
        return original_json(self, **kw)

    def spy_tools(self: LLM, **kw: Any) -> Any:
        prompts[kw["task"]] = kw["prompt"]
        return original_tools(self, **kw)

    monkeypatch.setattr(LLM, "json", spy_json)
    monkeypatch.setattr(LLM, "run_tools", spy_tools)
    plan_trip("5 days in Japan, need a hotel and flights", llm, profile=["vegetarian"])

    for task in ("destination_research", "itinerary", "accommodation", "transport"):
        assert "Always true for this traveller: vegetarian" in prompts[task], task


def test_no_profile_adds_nothing_to_prompts(llm: LLM, monkeypatch) -> None:
    prompts: list[str] = []
    original = LLM.json
    monkeypatch.setattr(
        LLM, "json", lambda self, **kw: prompts.append(kw["prompt"]) or original(self, **kw)
    )
    plan_trip("5 days in Japan", llm)
    assert not any("Always true" in p for p in prompts)


# -- API ----------------------------------------------------------------------


def test_profile_starts_empty_and_can_be_saved_and_forgotten(client, user) -> None:
    assert client.get(f"/api/profile/{user}").json() == {"preferences": []}

    saved = client.put(f"/api/profile/{user}", json={"preferences": ["Vegetarian", "kids"]})
    assert saved.json() == {"preferences": ["vegetarian", "kids"]}

    client.put(f"/api/profile/{user}", json={"preferences": ["kids"]})
    assert client.get(f"/api/profile/{user}").json() == {"preferences": ["kids"]}


def test_bad_user_ids_and_oversized_profiles_are_refused(client, user) -> None:
    assert client.get("/api/profile/x").status_code == 422
    assert client.get("/api/profile/../../etc").status_code in (404, 422)
    too_many = {"preferences": [f"p{i}" for i in range(MAX_PREFERENCES + 1)]}
    assert client.put(f"/api/profile/{user}", json=too_many).status_code == 422
    too_long = {"preferences": ["x" * 500]}
    assert client.put(f"/api/profile/{user}", json=too_long).status_code == 422


def test_a_plan_suggests_but_does_not_save(client, user) -> None:
    with client.websocket_connect("/ws/plan") as ws:
        ws.send_json({"request": "I'm vegan. 3 days in Lisbon", "provider": "mock", "user_id": user})
        while (message := ws.receive_json())["type"] not in ("done", "clarification", "error"):
            pass

    assert message["suggested_preferences"] == ["vegan"]
    assert client.get(f"/api/profile/{user}").json() == {"preferences": []}


def test_saved_preferences_are_applied_and_kept_with_the_trip(client, user) -> None:
    from backend.db import load_state

    client.put(f"/api/profile/{user}", json={"preferences": ["vegetarian"]})
    body = client.post(
        "/api/plan", json={"request": "3 days in Lisbon", "provider": "mock", "user_id": user}
    ).json()

    # Saved with the trip, so later edits plan with the same preferences.
    assert load_state(body["trip_id"])["profile"] == ["vegetarian"]


def test_no_user_id_means_no_suggestions(client) -> None:
    with client.websocket_connect("/ws/plan") as ws:
        ws.send_json({"request": "I'm vegan. 3 days in Lisbon", "provider": "mock"})
        while (message := ws.receive_json())["type"] not in ("done", "clarification", "error"):
            pass
    assert message["suggested_preferences"] == []
