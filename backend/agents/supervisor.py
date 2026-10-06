"""Supervisor agent — decides which workers run, and in what order.

The supervisor picks a subset of the four worker agents. The aggregator is not
part of that decision: it always runs last to format whatever the workers
produced, so it is appended by the graph and excluded from the routing metrics
in `eval/selection.py`.

Routing decisions are the thing the eval layer measures, so this node keeps the
decision explicit and inspectable: it writes an ordered `route_plan` into state
and the graph simply walks it.
"""

from __future__ import annotations

from typing import Any

from backend.llm import LLM, LLMError, register_mock
from backend.state import (
    ACCOMMODATION,
    AGGREGATOR,
    ITINERARY,
    RESEARCH,
    TRANSPORT,
    TravelState,
)

#: Canonical execution order. A worker may only run after everything it reads
#: from state has already run, so any valid plan is a subsequence of this.
CANONICAL_ORDER: tuple[str, ...] = (RESEARCH, ITINERARY, ACCOMMODATION, TRANSPORT)

#: What each worker needs to have available before it can do useful work.
DEPENDENCIES: dict[str, tuple[str, ...]] = {
    RESEARCH: (),
    ITINERARY: (RESEARCH,),
    ACCOMMODATION: (),
    TRANSPORT: (),
}

ROUTE_SYSTEM = f"""You are the supervisor of a travel planning team. You decide which
specialist agents are needed for a request — no more, no less. Invoking an agent whose
output the traveller did not ask for wastes time and clutters the answer.

Available agents:
  {RESEARCH}  - attractions, weather, practical/visa/safety notes for a place
  {ITINERARY}     - a day-by-day schedule; requires {RESEARCH} to have run
  {ACCOMMODATION}  - where to stay: neighbourhoods that suit the traveller and budget
  {TRANSPORT}     - how to arrive (airports, rail) and get around locally

Return JSON:
  {{"agents": [...], "reasoning": "one sentence"}}

Rules:
- "agents" is an ordered list drawn only from the four names above.
- Include an agent only if the request actually calls for its output.
- A question about a place (weather, what to see, is it safe) needs only {RESEARCH}.
- A multi-day trip plan needs {RESEARCH} and {ITINERARY}; add {ACCOMMODATION} and
  {TRANSPORT} when the traveller wants somewhere to stay or a way to get there.
- If a request mentions only hotels, {ACCOMMODATION} alone is correct.
- {ITINERARY} must come after {RESEARCH}."""

_ACCOMMODATION_WORDS = (
    "hotel", "hostel", "stay", "accommodation", "lodging", "airbnb", "where to sleep",
    "place to stay", "ryokan", "guesthouse", "resort",
)
_TRANSPORT_WORDS = (
    "flight", "fly", "train", "transport", "getting around", "get around", "rail",
    "bus", "car", "transfer", "airport", "how do i get",
)
_ITINERARY_WORDS = (
    "itinerary", "plan", "schedule", "day by day", "day-by-day", "trip", "days",
)
_RESEARCH_ONLY_WORDS = (
    "weather", "what to see", "attractions", "safe", "visa", "best time",
    "what's it like", "tell me about",
)


@register_mock("route_plan")
def _mock_route(context: dict[str, Any]) -> dict[str, Any]:
    """Keyword router standing in for the model when no API key is set."""
    text = str(context.get("request", "")).lower()
    duration = context.get("duration_days")

    agents: list[str] = []
    wants_trip = bool(duration) or any(w in text for w in _ITINERARY_WORDS)
    wants_stay = any(w in text for w in _ACCOMMODATION_WORDS)
    wants_transport = any(w in text for w in _TRANSPORT_WORDS)
    asks_about_place = any(w in text for w in _RESEARCH_ONLY_WORDS)

    if wants_trip or asks_about_place or not (wants_stay or wants_transport):
        agents.append(RESEARCH)
    if wants_trip:
        agents.append(ITINERARY)
    if wants_stay or (wants_trip and not asks_about_place):
        agents.append(ACCOMMODATION)
    if wants_transport or (wants_trip and not asks_about_place):
        agents.append(TRANSPORT)

    return {"agents": agents or [RESEARCH], "reasoning": "keyword routing (mock backend)"}


def _repair(
    agents: list[str], available: tuple[str, ...] = CANONICAL_ORDER
) -> tuple[list[str], list[str]]:
    """Drop unknown/duplicate agents and fix dependency-violating order.

    `available` is the set of workers the current graph actually has nodes for,
    so a partially built graph never routes to a node that does not exist.

    Returns the usable plan and a list of human-readable repairs, which the
    trajectory checks in `eval/` report on.
    """
    repairs: list[str] = []
    cleaned: list[str] = []
    for agent in agents:
        if agent not in available:
            repairs.append(f"dropped unavailable agent {agent!r}")
        elif agent in cleaned:
            repairs.append(f"dropped duplicate {agent!r}")
        else:
            cleaned.append(agent)

    for agent, needs in DEPENDENCIES.items():
        if agent in cleaned:
            for need in needs:
                if need not in cleaned and need in available:
                    cleaned.append(need)
                    repairs.append(f"added {need!r}, required by {agent!r}")

    ordered = sorted(cleaned, key=CANONICAL_ORDER.index)
    if ordered != cleaned:
        repairs.append("reordered plan to satisfy agent dependencies")

    return ordered, repairs


def supervisor(
    state: TravelState, llm: LLM, available: tuple[str, ...] = CANONICAL_ORDER
) -> dict[str, Any]:
    """Graph node: plan the route on the first visit, then advance through it."""
    if state.get("route_plan"):
        return {"cursor": state.get("cursor", 0) + 1}

    params = state.get("params")
    request = state["request"]

    try:
        decision = llm.json(
            task="route_plan",
            system=ROUTE_SYSTEM,
            prompt=(
                f"Request: {request}\n"
                f"Parsed parameters: {params.model_dump_json() if params else '{}'}"
            ),
            context={
                "request": request,
                "duration_days": params.duration_days if params else None,
            },
            max_tokens=512,
        )
        agents = list(decision.get("agents") or [])
        reasoning = str(decision.get("reasoning", ""))
    except LLMError as exc:
        agents, reasoning = list(CANONICAL_ORDER), f"routing failed ({exc}); ran everything"

    plan, repairs = _repair(agents, available)
    if not plan:
        plan, _ = _repair([RESEARCH], available)
        repairs.append("empty plan; defaulted to research")

    meta = dict(state.get("meta") or {})
    meta["routing"] = {
        "requested": agents,
        "plan": list(plan),
        "repairs": repairs,
        "reasoning": reasoning,
    }

    return {
        "route_plan": [*plan, AGGREGATOR],
        "cursor": 0,
        "meta": meta,
        "trace": ["supervisor"],
        "errors": [f"supervisor: {r}" for r in repairs],
    }
