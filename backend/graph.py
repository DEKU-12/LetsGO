"""LangGraph wiring.

    START -> parse_request (guard)
                 |-- unusable request --> END (with a clarifying question)
                 `-> supervisor <-> worker*  -> aggregator -> validate_plan -> END

The supervisor writes an ordered `route_plan` into state; every worker returns
to the supervisor, which advances the cursor. That loop is what makes the
trajectory recorded in `state["trace"]` a real record of what happened rather
than a fixed pipeline.
"""

from __future__ import annotations

from typing import Any, Callable

from langgraph.graph import END, START, StateGraph

from backend.agents.aggregator import aggregator
from backend.agents.destination_research import destination_research
from backend.agents.supervisor import supervisor
from backend.guards import parse_request, validate_plan
from backend.llm import LLM
from backend.state import AGGREGATOR, RESEARCH, TravelState, new_state

#: Worker nodes this graph can dispatch to, in canonical dependency order.
#: Phase 2 adds itinerary, accommodation and transport here.
WORKER_NODES: dict[str, Callable[[TravelState, LLM], dict[str, Any]]] = {
    RESEARCH: destination_research,
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
        graph.add_edge(name, "supervisor")
    graph.add_edge(AGGREGATOR, "validate_plan")
    graph.add_edge("validate_plan", END)

    return graph.compile()


def plan_trip(request: str, llm: LLM | None = None) -> TravelState:
    """Run one request end to end and return the final state."""
    llm = llm or LLM()
    return build_graph(llm).invoke(new_state(request))
