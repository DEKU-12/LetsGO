"""Bring-your-own-key: visitors' keys are used per request and never kept."""

from __future__ import annotations

import dataclasses
import json
import sqlite3

import pytest
from fastapi.testclient import TestClient

import backend.api
import backend.llm
from backend.api import app
from backend.llm import LLM, LLMError

FAKE_KEY = "sk-ant-visitor-0000000000000000"


@pytest.fixture
def deployed(monkeypatch):
    """Run as the deployed site: REQUIRE_USER_KEY=true."""
    on = dataclasses.replace(backend.llm.settings, require_user_key=True)
    monkeypatch.setattr(backend.llm, "settings", on)
    monkeypatch.setattr(backend.api, "settings", on)


def test_deployed_site_refuses_a_model_without_a_key(deployed) -> None:
    with pytest.raises(LLMError, match="your own API key"):
        LLM(provider="anthropic")
    with pytest.raises(LLMError):
        LLM()  # no silent fallback to the server's own key


def test_demo_mode_needs_no_key(deployed) -> None:
    assert LLM(provider="mock").is_mock


def test_the_visitors_key_reaches_the_provider_and_nowhere_else(deployed) -> None:
    llm = LLM(provider="anthropic", api_key=FAKE_KEY)
    assert llm._backend._client.api_key == FAKE_KEY
    assert FAKE_KEY not in repr(llm) and FAKE_KEY not in repr(llm._backend)


def test_health_says_a_key_is_needed(deployed) -> None:
    with TestClient(app) as client:
        assert client.get("/api/health").json()["require_user_key"] is True


def test_deployed_plan_without_a_key_is_a_400(deployed) -> None:
    with TestClient(app) as client:
        response = client.post("/api/plan", json={"request": "3 days in Rome",
                                                  "provider": "anthropic"})
    assert response.status_code == 400
    assert "API key" in response.json()["detail"]


def test_a_key_sent_with_a_plan_is_not_saved(monkeypatch) -> None:
    """The key travels with the request; the saved trip must not contain it."""
    from backend.config import settings

    with TestClient(app) as client:
        body = client.post("/api/plan", json={
            "request": "3 days in Lisbon", "provider": "mock", "api_key": FAKE_KEY,
        }).json()

    db = settings.database_url.removeprefix("sqlite:///")
    row = sqlite3.connect(db).execute(
        "SELECT * FROM trips WHERE id = ?", (body["trip_id"],)
    ).fetchone()
    assert FAKE_KEY not in json.dumps(row, default=str)
