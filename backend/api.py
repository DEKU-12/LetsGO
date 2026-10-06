"""HTTP and WebSocket surface.

    uv run uvicorn backend.api:app --reload

The WebSocket is the point. A full run takes most of a minute, and a multi-agent
system that shows nothing while it works is indistinguishable from a slow
chatbot. `/ws/plan` emits an event as each agent finishes, so the client can
show the team working.

`POST /api/plan` exists for scripts and tests that just want the answer.
"""

from __future__ import annotations

import logging
import re
import time
from contextlib import asynccontextmanager
from datetime import date
from typing import Annotated, Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from backend.adapters.weather import WeatherAdapter
from backend.config import PROJECT_ROOT
from backend.db import (
    get_profile,
    get_trip,
    init_db,
    list_trips,
    load_state,
    save_trip,
    set_profile,
)
from backend.edit import check_today, edit_trip, trip_day
from backend.graph import WORKER_NODES, plan_trip, stream_trip
from backend.llm import LLM, LLMError
from backend.preferences import MAX_LENGTH, MAX_PREFERENCES, clean, suggest_preferences
from backend.state import AGGREGATOR, TravelState

log = logging.getLogger(__name__)

#: What each node is doing, in words a person would use.
AGENT_LABELS: dict[str, str] = {
    "parse_request": "Reading your request",
    "supervisor": "Deciding which specialists are needed",
    "destination_research": "Researching the destination",
    "itinerary": "Building the day-by-day schedule",
    "check_itinerary": "Checking the schedule is realistic",
    "accommodation": "Finding places to stay",
    "transport": "Working out how to get there",
    AGGREGATOR: "Writing up the plan",
    "validate_plan": "Checking the result",
}

#: Long enough for a real trip on a slow model, short enough to fail visibly.
MAX_REQUEST_CHARS = 2000

#: The anonymous id a browser makes for itself (a UUID). Anything else is
#: refused, so the id cannot be used to probe or stuff the profiles table.
USER_ID = re.compile(r"^[A-Za-z0-9-]{8,64}$")


class PlanRequest(BaseModel):
    request: str = Field(min_length=1, max_length=MAX_REQUEST_CHARS)
    provider: str | None = None
    #: Anonymous browser id; its saved preferences are applied to the plan.
    user_id: str | None = Field(default=None, pattern=USER_ID.pattern)


class ProfileUpdate(BaseModel):
    preferences: list[Annotated[str, Field(max_length=MAX_LENGTH)]] = Field(
        max_length=MAX_PREFERENCES
    )


class EditRequest(BaseModel):
    message: str = Field(min_length=1, max_length=MAX_REQUEST_CHARS)
    #: The day of the trip the traveller is on, if they said.
    today: int | None = Field(default=None, ge=1)
    provider: str | None = None


class TodayRequest(BaseModel):
    today: int | None = Field(default=None, ge=1)
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
        "days": len(state["itinerary"].days) if state.get("itinerary") else 0,
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

    profile = await run_in_threadpool(get_profile, body.user_id) if body.user_id else []
    started = time.monotonic()
    state = await run_in_threadpool(plan_trip, body.request, llm, profile)
    elapsed = time.monotonic() - started

    trip_id = await run_in_threadpool(
        save_trip, state, provider=llm.provider, model=llm.model, duration_s=elapsed
    )
    return _result(state, trip_id)


@app.get("/api/profile/{user_id}")
def read_profile(user_id: str) -> dict[str, Any]:
    """The preferences this browser asked us to remember."""
    _check_user_id(user_id)
    return {"preferences": get_profile(user_id)}


@app.put("/api/profile/{user_id}")
def update_profile(user_id: str, body: ProfileUpdate) -> dict[str, Any]:
    """Replace the saved preferences: used both to remember one and to forget one."""
    _check_user_id(user_id)
    return {"preferences": set_profile(user_id, clean(body.preferences))}


def _check_user_id(user_id: str) -> None:
    if not USER_ID.match(user_id):
        raise HTTPException(status_code=422, detail="invalid user id")


@app.get("/api/trips")
def trips(limit: int = 50) -> list[dict[str, Any]]:
    return list_trips(limit=limit)


@app.get("/api/trips/{trip_id}")
def trip(trip_id: int) -> dict[str, Any]:
    found = get_trip(trip_id)
    if found is None:
        raise HTTPException(status_code=404, detail="no such trip")
    return found


@app.post("/api/trips/{trip_id}/edit")
async def edit_plan(trip_id: int, body: EditRequest) -> dict[str, Any]:
    """Change a saved plan — "less walking on day 3".

    Saves the result as a new trip whose ``parent_id`` is this one, so the old
    version stays put and undo is just going back to it. A message that is not
    an edit this plan can take returns ``type: "reply"`` and saves nothing.

    ``today`` (the day of the trip the traveller is on) lets "rearrange today"
    reach the right day; without it, the plan's start date is used if it has
    one. When today is known, the live weather is passed along too.
    """
    state, llm = await _editable(trip_id, body.provider)
    today = body.today or trip_day(state, date.today())
    conditions = await _weather_now(state) if today else None

    started = time.monotonic()
    edited, reply = await run_in_threadpool(
        edit_trip, state, body.message, llm, today, conditions
    )
    return await _edit_result(trip_id, edited, reply, llm, started, body.message)


@app.post("/api/trips/{trip_id}/today")
async def help_today(trip_id: int, body: TodayRequest) -> dict[str, Any]:
    """Trip-day help: check today's weather and rearrange today if it is bad.

    Changes nothing, and says so, when the weather is fine. Needs to know which
    day of the trip it is: ``today`` in the body, or the plan's start date.
    """
    state, llm = await _editable(trip_id, body.provider)
    today = body.today or trip_day(state, date.today())
    if today is None:
        raise HTTPException(status_code=400, detail="Which day of the trip are you on?")
    days = len(state["itinerary"].days) if state.get("itinerary") else 0
    if not 1 <= today <= days:
        raise HTTPException(status_code=400, detail=f"This plan has {days} days.")

    conditions = await _weather_now(state)
    started = time.monotonic()
    edited, reply = await run_in_threadpool(check_today, state, today, conditions, llm)
    result = await _edit_result(trip_id, edited, reply, llm, started, f"Check today (day {today})")
    result["weather"] = conditions
    return result


async def _editable(trip_id: int, provider: str | None) -> tuple[TravelState, LLM]:
    state = await run_in_threadpool(load_state, trip_id)
    if state is None:
        raise HTTPException(status_code=404, detail="no such trip")
    if not state.get("final_plan"):
        raise HTTPException(status_code=400, detail="this trip has no plan to edit")
    try:
        return state, LLM(provider=provider)
    except LLMError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


async def _weather_now(state: TravelState) -> dict[str, Any]:
    result = await run_in_threadpool(
        WeatherAdapter().fetch, destination=state["params"].destination, when="now"
    )
    return result.data


async def _edit_result(
    trip_id: int,
    edited: TravelState | None,
    reply: str,
    llm: LLM,
    started: float,
    message: str,
) -> dict[str, Any]:
    """Save an edited plan as a new version, or pass the reply through."""
    if edited is None:
        return {"type": "reply", "reply": reply, "trip_id": trip_id}

    elapsed = time.monotonic() - started
    new_id = await run_in_threadpool(
        save_trip, edited, provider=llm.provider, model=llm.model, duration_s=elapsed,
        parent_id=trip_id, edit_message=message,
    )
    edit = (edited.get("meta") or {}).get("edit", {})
    return {
        **_result(edited, new_id),
        "type": "done",
        "parent_id": trip_id,
        "edited": {
            "section": edit.get("agent"),
            "days": edit.get("days", []),
            "weather": edit.get("weather"),
            "changed": edit.get("changed", True),
        },
        "duration_s": round(elapsed, 1),
    }


@app.websocket("/ws/plan")
async def plan_socket(websocket: WebSocket) -> None:
    """Stream a run, one event per agent.

    Protocol — client sends ``{"request": "...", "provider": null}``, then
    receives ``started``, any number of ``route``/``agent`` events, then exactly
    one of ``done``, ``clarification`` or ``error``. After ``done``, a
    ``suggestions`` message may follow with preferences to offer remembering.
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

    user_id = payload.get("user_id")
    user_id = user_id if isinstance(user_id, str) and USER_ID.match(user_id) else None
    profile = await run_in_threadpool(get_profile, user_id) if user_id else []

    await websocket.send_json({
        "type": "started",
        "provider": llm.provider,
        "model": llm.model,
        "mock": llm.is_mock,
    })

    started = time.monotonic()
    state: dict[str, Any] = {}

    try:
        stream = stream_trip(request, llm, profile)
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
    result["profile"] = profile

    # The plan goes out the moment it is ready; nothing optional holds it up.
    await websocket.send_json(result)

    # Then offer to remember lasting preferences from this request, as a
    # separate message. Never saved here: the traveller says yes to each one.
    # A failure, or the client leaving first, only loses the offer.
    if user_id and not state.get("clarification"):
        try:
            suggestions = await run_in_threadpool(suggest_preferences, request, llm, profile)
            if suggestions:
                await websocket.send_json({"type": "suggestions", "preferences": suggestions})
        except LLMError as exc:
            log.warning("preference suggestions skipped: %s", exc)
        except WebSocketDisconnect:
            return
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
