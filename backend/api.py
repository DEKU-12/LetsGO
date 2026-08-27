"""HTTP and WebSocket surface.

    uv run uvicorn backend.api:app --reload

The WebSocket is the point. A full run takes most of a minute, and a multi-agent
system that shows nothing while it works is indistinguishable from a slow
chatbot. `/ws/plan` emits an event as each agent finishes, so the client can
show the team working.

`POST /api/plan` exists for scripts and tests that just want the answer.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from backend.config import PROJECT_ROOT
from backend.db import get_trip, init_db, list_trips, save_trip
from backend.graph import WORKER_NODES, plan_trip, stream_trip
from backend.llm import LLM, LLMError
from backend.state import AGGREGATOR

#: What each node is doing, in words a person would use.
AGENT_LABELS: dict[str, str] = {
    "parse_request": "Reading your request",
    "supervisor": "Deciding which specialists are needed",
    "destination_research": "Researching the destination",
    "itinerary": "Building the day-by-day schedule",
    "accommodation": "Finding places to stay",
    "transport": "Working out how to get there",
    AGGREGATOR: "Writing up the plan",
    "validate_plan": "Checking the result",
}

#: Long enough for a real trip on a slow model, short enough to fail visibly.
MAX_REQUEST_CHARS = 2000


class PlanRequest(BaseModel):
    request: str = Field(min_length=1, max_length=MAX_REQUEST_CHARS)
    provider: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="LetsGO", version="0.1.0", lifespan=lifespan)

# The Vite dev server runs on another port; in production the UI is built and
# served as static files, so this only matters for local development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _label(node: str) -> str:
    return AGENT_LABELS.get(node, node.replace("_", " ").capitalize())


def _result(state: dict[str, Any], trip_id: int | None) -> dict[str, Any]:
    params = state.get("params")
    return {
        "trip_id": trip_id,
        "plan": state.get("final_plan"),
        "clarification": state.get("clarification"),
        "destination": params.destination if params else None,
        "route": [a for a in (state.get("route_plan") or [])],
        "trace": state.get("trace") or [],
        "sources": sorted(set(state.get("sources") or [])),
        "notes": state.get("errors") or [],
    }


@app.get("/api/health")
def health() -> dict[str, Any]:
    llm = LLM()
    return {
        "status": "ok",
        "provider": llm.provider,
        "model": llm.model,
        "mock": llm.is_mock,
        "agents": list(WORKER_NODES),
    }


@app.post("/api/plan")
async def create_plan(body: PlanRequest) -> dict[str, Any]:
    """Plan a trip and wait for the answer. Use the WebSocket for progress."""
    try:
        llm = LLM(provider=body.provider)
    except LLMError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    started = time.monotonic()
    state = await run_in_threadpool(plan_trip, body.request, llm)
    elapsed = time.monotonic() - started

    trip_id = await run_in_threadpool(
        save_trip, state, provider=llm.provider, model=llm.model, duration_s=elapsed
    )
    return _result(state, trip_id)


@app.get("/api/trips")
def trips(limit: int = 50) -> list[dict[str, Any]]:
    return list_trips(limit=limit)


@app.get("/api/trips/{trip_id}")
def trip(trip_id: int) -> dict[str, Any]:
    found = get_trip(trip_id)
    if found is None:
        raise HTTPException(status_code=404, detail="no such trip")
    return found


@app.websocket("/ws/plan")
async def plan_socket(websocket: WebSocket) -> None:
    """Stream a run, one event per agent.

    Protocol — client sends ``{"request": "...", "provider": null}``, then
    receives ``started``, any number of ``route``/``agent`` events, and finally
    exactly one of ``done``, ``clarification`` or ``error``.
    """
    await websocket.accept()

    try:
        payload = await websocket.receive_json()
    except (WebSocketDisconnect, ValueError):
        return

    request = str(payload.get("request") or "").strip()
    if not request:
        await websocket.send_json({"type": "error", "message": "empty request"})
        await websocket.close()
        return
    if len(request) > MAX_REQUEST_CHARS:
        await websocket.send_json({"type": "error", "message": "request too long"})
        await websocket.close()
        return

    try:
        llm = LLM(provider=payload.get("provider"))
    except LLMError as exc:
        await websocket.send_json({"type": "error", "message": str(exc)})
        await websocket.close()
        return

    await websocket.send_json({
        "type": "started",
        "provider": llm.provider,
        "model": llm.model,
        "mock": llm.is_mock,
    })

    started = time.monotonic()
    state: dict[str, Any] = {}

    try:
        stream = stream_trip(request, llm)
        while True:
            step = await run_in_threadpool(lambda: next(stream, None))
            if step is None:
                break
            node, update, state = step

            if node == "supervisor" and "route_plan" in update:
                routing = (update.get("meta") or {}).get("routing", {})
                await websocket.send_json({
                    "type": "route",
                    "plan": [
                        {"agent": a, "label": _label(a)} for a in update["route_plan"]
                    ],
                    "reasoning": routing.get("reasoning", ""),
                })
            elif node != "supervisor":
                await websocket.send_json({
                    "type": "agent",
                    "agent": node,
                    "label": _label(node),
                    "notes": [e for e in (update.get("errors") or [])],
                })
    except WebSocketDisconnect:
        return
    except Exception as exc:  # noqa: BLE001 - the socket must always be told
        await websocket.send_json({"type": "error", "message": str(exc)})
        await websocket.close()
        return

    elapsed = time.monotonic() - started
    trip_id = await run_in_threadpool(
        save_trip, state, provider=llm.provider, model=llm.model, duration_s=elapsed
    )

    result = _result(state, trip_id)
    result["duration_s"] = round(elapsed, 1)
    result["type"] = "clarification" if state.get("clarification") else "done"

    await websocket.send_json(result)
    await websocket.close()


# ---------------------------------------------------------------------------
# The built UI, if there is one
# ---------------------------------------------------------------------------
#
# In development the Vite dev server runs separately and proxies /api and /ws
# here. After `npm --prefix frontend run build` this serves the built files too,
# so the whole thing is one process and one port — which is what makes a
# one-command demo possible.

_DIST = PROJECT_ROOT / "frontend" / "dist"

if _DIST.is_dir():
    app.mount("/assets", StaticFiles(directory=_DIST / "assets"), name="assets")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(_DIST / "index.html")
