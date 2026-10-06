
from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Callable

from langgraph.graph import END, START, StateGraph

from backend.agents.accommodation import accommodation
from backend.agents.aggregator import aggregator
from backend.agents.check import after_check, check_itinerary
from backend.agents.destination_research import destination_research
from backend.agents.itinerary import itinerary
from backend.agents.supervisor import supervisor
from backend.agents.transport import transport
from backend.guards import parse_request, validate_plan
from backend.llm import LLM
from backend.state import (
    ACCOMMODATION,
    AGGREGATOR,
    ITINERARY,
    RESEARCH,
    TRANSPORT,
    TravelState,
    new_state,
)

#: Worker nodes this graph can dispatch to, in canonical dependency order.
#: The supervisor may route to any subset of these; the aggregator always runs.
WORKER_NODES: dict[str, Callable[[TravelState, LLM], dict[str, Any]]] = {
    RESEARCH: destination_research,
    ITINERARY: itinerary,
    ACCOMMODATION: accommodation,
    TRANSPORT: transport,
}


def _after_guard(state: TravelState) -> str:
    """Stop early if the request could not be parsed into a real trip."""
    return END if state.get("clarification") else "supervisor"


def _route_next(state: TravelState) -> str:
    """Dispatch to the next agent in the supervisor's plan."""
    plan = state.get("route_plan") or []
    cursor = state.get("cursor", 0)
    if cursor >= len(plan):
        return END
    return plan[cursor]


def build_graph(llm: LLM | None = None):
    """Compile the travel planning graph."""
    llm = llm or LLM()
    available = tuple(WORKER_NODES)

    graph = StateGraph(TravelState)

    graph.add_node("parse_request", lambda s: parse_request(s, llm))
    graph.add_node("supervisor", lambda s: supervisor(s, llm, available))
    for name, fn in WORKER_NODES.items():
        graph.add_node(name, lambda s, _fn=fn: _fn(s, llm))
    graph.add_node("check_itinerary", check_itinerary)
    graph.add_node(AGGREGATOR, lambda s: aggregator(s, llm))
    graph.add_node("validate_plan", validate_plan)

    graph.add_edge(START, "parse_request")
    graph.add_conditional_edges(
        "parse_request", _after_guard, {"supervisor": "supervisor", END: END}
    )
    graph.add_conditional_edges(
        "supervisor",
        _route_next,
        {**{name: name for name in WORKER_NODES}, AGGREGATOR: AGGREGATOR, END: END},
    )
    for name in WORKER_NODES:
        if name != ITINERARY:
            graph.add_edge(name, "supervisor")
    # The itinerary is checked before the supervisor moves on, and sent back
    # with a list of problems if it fails.
    graph.add_edge(ITINERARY, "check_itinerary")
    graph.add_conditional_edges(
        "check_itinerary", after_check, {ITINERARY: ITINERARY, "supervisor": "supervisor"}
    )
    graph.add_edge(AGGREGATOR, "validate_plan")
    graph.add_edge("validate_plan", END)

    return graph.compile()


def plan_trip(request: str, llm: LLM | None = None) -> TravelState:
    """Run one request end to end and return the final state."""
    llm = llm or LLM()
    return build_graph(llm).invoke(new_state(request))


def stream_trip(
    request: str, llm: LLM | None = None
) -> Iterator[tuple[str, dict[str, Any], TravelState]]:
    """Run a request, yielding after each node so callers can show progress.

    Yields ``(node_name, that_node's_update, state_so_far)``. A full run takes
    the better part of a minute; without this the caller has nothing to show
    until it finishes, and a multi-agent system that reveals nothing while it
    works is indistinguishable from a slow one.
    """
    llm = llm or LLM()
    state: TravelState = new_state(request)

    for step in build_graph(llm).stream(state, stream_mode="updates"):
        for node, update in step.items():
            if not isinstance(update, dict):
                continue
            apply_update(state, update)
            yield node, update, state


def apply_update(state: TravelState, update: dict[str, Any]) -> TravelState:
    """Fold one node's update into state, the way the graph would.

    `trace`, `errors` and `sources` accumulate; everything else replaces. This
    mirrors the reducers declared on TravelState, for code that runs nodes
    outside the graph (streaming, plan edits).
    """
    for key, value in update.items():
        if key in {"trace", "errors", "sources"}:
            state[key] = [*(state.get(key) or []), *value]
        else:
            state[key] = value
    return state
