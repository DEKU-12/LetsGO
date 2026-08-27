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


def test_full_trip_routes_through_every_worker(llm: LLM) -> None:
    state = plan_trip(
        "5 days in Japan, mid-range budget, love food and history, need hotels and flights",
        llm,
    )
    trace = state["trace"]

    for worker in ("destination_research", "itinerary", "accommodation", "transport"):
        assert worker in trace, f"{worker} never ran"

    # Dependencies: the itinerary cannot be built before the research exists.
    assert trace.index("destination_research") < trace.index("itinerary")
    assert trace.index("aggregator") == len(trace) - 2


def test_plan_labels_data_that_came_from_mocks(llm: LLM) -> None:
    state = plan_trip(
        "5 days in Japan, mid-range budget, need hotels and flights", llm
    )
    plan = state["final_plan"]

    assert "lodging:mock" in state["sources"]
    assert "not bookable" in plan
    assert "not quotes" in plan


def test_weather_only_question_does_not_book_a_hotel(llm: LLM) -> None:
    state = plan_trip("What's the weather like in Rome?", llm)

    assert "accommodation" not in state["route_plan"]
    assert "transport" not in state["route_plan"]
    assert state["accommodation"] is None


def test_information_questions_are_answered_not_refused(llm: LLM) -> None:
    """A question about a place is a valid request, not something to clarify.

    Caught by eval/run.py: the guard was rejecting these, so the supervisor
    never got the chance to route them to research.
    """
    for question in (
        "What's the weather like in Rome in October?",
        "Is Marrakesh safe for a solo female traveller?",
    ):
        state = plan_trip(question, llm)
        assert state["clarification"] is None, f"refused: {question}"
        assert RESEARCH in state["trace"]
