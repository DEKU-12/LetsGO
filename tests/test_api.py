"""The HTTP and WebSocket surface, on the mock backend."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.api import app


@pytest.fixture
def client() -> TestClient:
    with TestClient(app) as c:
        yield c


def test_health_reports_the_backend(client: TestClient) -> None:
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert set(body["agents"]) == {
        "destination_research", "itinerary", "accommodation", "transport"
    }


def test_plan_endpoint_returns_a_plan(client: TestClient) -> None:
    body = client.post(
        "/api/plan",
        json={"request": "3 days in Lisbon, need a hotel", "provider": "mock"},
    ).json()

    assert body["plan"]
    assert body["destination"] == "Lisbon"
    assert body["trip_id"]
    assert "aggregator" in body["route"]


def test_empty_request_is_rejected(client: TestClient) -> None:
    assert client.post("/api/plan", json={"request": ""}).status_code == 422


def test_overlong_request_is_rejected(client: TestClient) -> None:
    response = client.post("/api/plan", json={"request": "x" * 5000})
    assert response.status_code == 422


def test_a_planned_trip_is_saved_and_retrievable(client: TestClient) -> None:
    created = client.post(
        "/api/plan",
        json={"request": "3 days in Lisbon, need a hotel", "provider": "mock"},
    ).json()

    detail = client.get(f"/api/trips/{created['trip_id']}").json()
    assert detail["request"].startswith("3 days in Lisbon")
    assert detail["plan"] == created["plan"]
    assert [r["agent"] for r in detail["runs"]]

    assert any(t["id"] == created["trip_id"] for t in client.get("/api/trips").json())


def test_unknown_trip_is_a_404(client: TestClient) -> None:
    assert client.get("/api/trips/999999").status_code == 404


# -- websocket --------------------------------------------------------------


def test_websocket_streams_progress_then_the_plan(client: TestClient) -> None:
    events = []
    with client.websocket_connect("/ws/plan") as ws:
        ws.send_json({"request": "5 days in Japan, hotels and flights", "provider": "mock"})
        while True:
            message = ws.receive_json()
            events.append(message)
            if message["type"] in {"done", "clarification", "error"}:
                break

    kinds = [e["type"] for e in events]
    assert kinds[0] == "started"
    assert "route" in kinds, "the supervisor's decision was never sent"
    assert kinds[-1] == "done"

    # Progress must arrive before the answer, or the stream is pointless.
    assert kinds.index("agent") < len(kinds) - 1

    finished = {e["agent"] for e in events if e["type"] == "agent"}
    assert {"destination_research", "itinerary", "accommodation", "transport"} <= finished
    assert events[-1]["plan"]


def test_websocket_reports_a_clarification_rather_than_a_plan(client: TestClient) -> None:
    with client.websocket_connect("/ws/plan") as ws:
        ws.send_json({"request": "what is 2 + 2", "provider": "mock"})
        while True:
            message = ws.receive_json()
            if message["type"] in {"done", "clarification", "error"}:
                break

    assert message["type"] == "clarification"
    assert message["clarification"]
    assert message["plan"] is None


def test_websocket_rejects_an_empty_request(client: TestClient) -> None:
    with client.websocket_connect("/ws/plan") as ws:
        ws.send_json({"request": "   "})
        assert ws.receive_json()["type"] == "error"
