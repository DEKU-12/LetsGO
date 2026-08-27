"""Graph wiring: does a request actually flow through the nodes we think it does."""

from __future__ import annotations

import pytest

from backend.graph import WORKER_NODES, build_graph, plan_trip
from backend.llm import LLM
from backend.state import AGGREGATOR, RESEARCH


@pytest.fixture
def llm() -> LLM:
    return LLM(provider="mock")


def test_graph_compiles(llm: LLM) -> None:
    assert build_graph(llm) is not None


def test_end_to_end_produces_a_plan(llm: LLM) -> None:
    state = plan_trip("5 days in Japan, mid-range budget, love food and history", llm)

    assert state["params"] is not None
    assert state["params"].destination == "Japan"
    assert state["params"].duration_days == 5
    assert state["params"].budget == "mid-range"
    assert set(state["params"].preferences) >= {"food", "history"}

    assert state["research"] is not None
    assert state["research"].attractions
    assert state["research"].weather is not None

    assert state["final_plan"]
    assert state["final_plan"].startswith("# Japan")


def test_trace_records_the_path_taken(llm: LLM) -> None:
    state = plan_trip("7 day trip to Italy, love art and food", llm)
    trace = state["trace"]

    assert trace[0] == "guard:parse_request"
    assert "supervisor" in trace
    assert trace.index(RESEARCH) < trace.index(AGGREGATOR)
    assert trace[-1] == "guard:validate_plan"


def test_supervisor_writes_an_ordered_route_plan(llm: LLM) -> None:
    state = plan_trip("What's the weather like in Rome?", llm)

    plan = state["route_plan"]
    assert plan[-1] == AGGREGATOR
    assert RESEARCH in plan
    # Routing metrics only ever cover implemented workers.
    assert set(plan) <= set(WORKER_NODES) | {AGGREGATOR}


def test_nonsense_request_asks_for_clarification_instead_of_planning(llm: LLM) -> None:
    state = plan_trip("what is 2 + 2", llm)

    assert state["clarification"]
    assert state["final_plan"] is None
    assert RESEARCH not in state["trace"]


def test_output_guard_flags_a_missing_plan() -> None:
    from backend.guards import validate_plan

    result = validate_plan({"final_plan": ""})
    assert any("final plan is missing" in e for e in result["errors"])
