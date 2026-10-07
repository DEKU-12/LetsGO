"""The evaluation layer itself — metrics have to be right before they mean anything."""

from __future__ import annotations

import pytest

from backend.state import ACCOMMODATION, ITINERARY, RESEARCH, TRANSPORT
from eval.dataset import ROUTING_CASES, HumanGrade
from eval.judge import RUBRIC, judge_plan, validate_judge
from eval.selection import AgentScore, CaseResult, SelectionReport, evaluate_selection
from eval.trajectory import check_trajectory
from backend.llm import LLM


@pytest.fixture
def llm() -> LLM:
    return LLM(provider="mock")


# -- dataset ----------------------------------------------------------------


def test_dataset_covers_every_agent_and_the_empty_case() -> None:
    labelled = {a for case in ROUTING_CASES for a in case.expected}
    assert labelled == {RESEARCH, ITINERARY, ACCOMMODATION, TRANSPORT}
    assert any(case.expected == frozenset() for case in ROUTING_CASES)
    assert len(ROUTING_CASES) >= 15


def test_every_case_has_a_stated_reason() -> None:
    """A label without a rationale cannot be argued with, so it cannot be trusted."""
    for case in ROUTING_CASES:
        assert case.why.strip(), case.id


def test_itinerary_labels_always_imply_research() -> None:
    for case in ROUTING_CASES:
        if ITINERARY in case.expected:
            assert RESEARCH in case.expected, case.id


def test_case_ids_are_unique() -> None:
    ids = [c.id for c in ROUTING_CASES]
    assert len(ids) == len(set(ids))


# -- selection metrics ------------------------------------------------------


def test_precision_and_recall_arithmetic() -> None:
    score = AgentScore("x", tp=3, fp=1, fn=2)
    assert score.precision == pytest.approx(0.75)
    assert score.recall == pytest.approx(0.6)
    assert score.f1 == pytest.approx(2 * 0.75 * 0.6 / 1.35)


def test_a_perfect_run_scores_one() -> None:
    cases = ROUTING_CASES[:3]
    results = [CaseResult(c, c.expected, False) for c in cases]
    report = SelectionReport(results=results, per_agent={})
    assert report.exact_match == 1.0


def test_selection_runs_end_to_end_on_mock(llm: LLM) -> None:
    report = evaluate_selection(llm)
    assert len(report.results) == len(ROUTING_CASES)
    assert 0.0 <= report.exact_match <= 1.0
    assert 0.0 <= report.micro.f1 <= 1.0


# -- trajectory -------------------------------------------------------------


def test_trajectory_flags_a_dependency_violation() -> None:
    report = check_trajectory({"trace": [ITINERARY, RESEARCH], "final_plan": ""})
    assert any("ran before" in issue for issue in report.invalid_order)


def test_trajectory_flags_an_agent_that_produced_nothing() -> None:
    report = check_trajectory({"trace": [RESEARCH], "research": None, "final_plan": "x"})
    assert any("produced nothing" in issue for issue in report.unused_output)


# -- judge ------------------------------------------------------------------


def test_judge_marks_missing_sections_not_applicable(llm: LLM) -> None:
    """A research-only answer must not be marked down for having no schedule."""
    plan = "# Rome\n\n## The destination\n\n" + "detail " * 100
    result = judge_plan("x", "What's the weather in Rome?", plan, llm)

    assert "coherence" in result.not_applicable
    assert "coherence" not in result.scores


def test_judge_refuses_to_score_an_empty_plan(llm: LLM) -> None:
    assert judge_plan("x", "req", "", llm).error


def test_validation_reports_honestly_when_there_are_no_human_grades(llm: LLM) -> None:
    report = validate_judge(llm, grades=[])
    assert not report.usable
    assert report.n == 0
    assert "no human grades" in report.note


def test_validation_measures_error_against_human_scores(llm: LLM) -> None:
    plan = "# Trip\n\n## The destination\n\n## Day by day\n\n" + "detail " * 100
    grades = [
        HumanGrade(case_id=f"c{i}", request="5 days in Japan", plan=plan,
                   scores={k: 3 for k in RUBRIC})
        for i in range(5)
    ]
    report = validate_judge(llm, grades=grades)

    assert report.n == 5
    assert report.usable
    assert report.mae >= 0.0
    assert 0.0 <= report.within_one <= 1.0


def test_schedule_entries_that_are_not_researched_places_are_listed() -> None:
    from backend.state import (
        ActivityBlock,
        Attraction,
        ItineraryDay,
        ItineraryOutput,
        ResearchOutput,
    )
    from eval.trajectory import check_trajectory

    state = {
        "research": ResearchOutput(destination="Paris", attractions=[Attraction(name="Louvre")]),
        "itinerary": ItineraryOutput(days=[ItineraryDay(day=1, blocks=[
            ActivityBlock(time="09:00", title="Louvre"),
            ActivityBlock(time="13:00", title="Lunch nearby"),
            ActivityBlock(time="15:00", title="Musée d'Orsay"),  # never researched
        ])]),
    }
    assert check_trajectory(state).unresearched_stops == ["Musée d'Orsay"]
