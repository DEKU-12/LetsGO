"""LLM-as-judge, and the validation that makes it worth reading.

Scoring a plan with a model is easy. The hard question is whether that model's
scores mean anything, and the only way to answer it is to compare them against
scores a person assigned to the same plans.

So this module has two halves:

* :func:`judge_plan` — score one plan against the rubric. The judge must write
  its reasoning before its numbers; asking for a score first invites a number
  with a justification reverse-engineered around it.
* :func:`validate_judge` — compare the judge against human grades and report
  mean absolute error, agreement within one point, and correlation.

If no human grades exist, validation says so and reports nothing. An
unvalidated judge with a confident number attached is worse than no judge.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any

from backend.llm import LLM, LLMError, register_mock
from eval.dataset import HumanGrade, load_human_grades

#: The rubric. Each dimension is scored 1-5.
RUBRIC: dict[str, str] = {
    "completeness": "Does the plan cover everything the traveller asked for, and "
                    "nothing they explicitly said was already handled?",
    "budget": "Do the recommendations match the stated budget level? If no budget "
              "was stated, does the plan avoid assuming an expensive one?",
    "preferences": "Does the plan reflect the traveller's stated interests, rather "
                   "than being a generic list for the destination?",
    "coherence": "Is the day-by-day schedule realistic — sensible pacing, stops "
                 "grouped geographically, no impossible travel between them?",
    "realism": "Are the places, times and prices plausible? Would a person "
               "following this plan find it works?",
}

#: A judge only barely more useful than a coin flip is not worth calling.
MIN_GRADES_FOR_VALIDATION = 5

JUDGE_SYSTEM = f"""You grade travel plans against a fixed rubric. You are strict:
a 5 means a professional travel planner would sign their name to it, and a 3 means
usable but unremarkable. Most plans are not 5s.

Score each of these 1-5:
{chr(10).join(f"  {name}: {question}" for name, question in RUBRIC.items())}

Return JSON:
  {{"reasoning": {{"<dimension>": "one sentence of evidence from the plan"}},
    "scores": {{"<dimension>": <1-5, or the string "n/a">}}}}

Write the reasoning for a dimension before its score, and cite something specific
from the plan. Do not reward length. A short plan that answers the request beats a
long one that wanders.

Use "n/a" only when the traveller never asked for the thing that dimension
measures — coherence on a plan with no schedule, budget on a request that named
no budget and needed no prices. "n/a" is not an escape from a low score: if the
traveller asked for it and the plan did it badly, score it low."""


@register_mock("judge_plan")
def _mock_judge(context: dict[str, Any]) -> dict[str, Any]:
    """Structural stand-in for the no-API-key path.

    Deliberately crude: it counts sections and length rather than reading the
    plan, so nobody mistakes a mock run for a real evaluation.
    """
    plan = str(context.get("plan", ""))
    sections = plan.count("## ")
    length = len(plan)

    base = 2 + min(sections, 3)
    if length < 400:
        base -= 1
    base = max(1, min(base, 5))

    scores: dict[str, Any] = {k: base for k in RUBRIC}
    if "## Day by day" not in plan:
        scores["coherence"] = "n/a"
    if "## Where to stay" not in plan:
        scores["budget"] = "n/a"

    return {
        "reasoning": {k: "mock judge: structural heuristic, not a reading" for k in RUBRIC},
        "scores": scores,
    }


@dataclass
class JudgeResult:
    case_id: str
    scores: dict[str, int]
    reasoning: dict[str, str] = field(default_factory=dict)
    #: Dimensions the request never called for, excluded from the average
    #: rather than scored low.
    not_applicable: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def overall(self) -> float:
        return sum(self.scores.values()) / len(self.scores) if self.scores else 0.0


def judge_plan(case_id: str, request: str, plan: str, llm: LLM) -> JudgeResult:
    """Score one plan against the rubric."""
    if not plan.strip():
        return JudgeResult(case_id, {}, error="no plan to judge")

    try:
        raw = llm.json(
            task="judge_plan",
            system=JUDGE_SYSTEM,
            prompt=f"Traveller's request:\n{request}\n\nThe plan:\n{plan}",
            context={"plan": plan, "request": request},
            max_tokens=2048,
        )
    except LLMError as exc:
        return JudgeResult(case_id, {}, error=str(exc))

    scores: dict[str, int] = {}
    skipped: list[str] = []
    for dimension in RUBRIC:
        value = (raw.get("scores") or {}).get(dimension)
        if isinstance(value, str) and value.strip().lower() in {"n/a", "na", "none"}:
            # The request never asked for what this dimension measures.
            skipped.append(dimension)
            continue
        try:
            scores[dimension] = max(1, min(5, int(value)))
        except (TypeError, ValueError):
            continue

    return JudgeResult(
        case_id=case_id,
        scores=scores,
        reasoning={k: str(v) for k, v in (raw.get("reasoning") or {}).items()},
        not_applicable=skipped,
        error=None if scores else "judge returned no usable scores",
    )


# ---------------------------------------------------------------------------
# Validation — the number that decides whether any of the above is credible
# ---------------------------------------------------------------------------


@dataclass
class ValidationReport:
    pairs: list[tuple[float, float]]  # (human overall, judge overall)
    per_dimension_mae: dict[str, float] = field(default_factory=dict)
    note: str = ""

    @property
    def n(self) -> int:
        return len(self.pairs)

    @property
    def usable(self) -> bool:
        return self.n >= MIN_GRADES_FOR_VALIDATION

    @property
    def mae(self) -> float:
        """Mean absolute error, in rubric points."""
        if not self.pairs:
            return 0.0
        return sum(abs(h - j) for h, j in self.pairs) / len(self.pairs)

    @property
    def within_one(self) -> float:
        """Share of plans where judge and human agree to within one point."""
        if not self.pairs:
            return 0.0
        return sum(abs(h - j) <= 1.0 for h, j in self.pairs) / len(self.pairs)

    @property
    def correlation(self) -> float | None:
        """Pearson r. None when there is not enough spread to compute one."""
        if self.n < 3:
            return None
        human = [h for h, _ in self.pairs]
        judge = [j for _, j in self.pairs]
        if len(set(human)) < 2 or len(set(judge)) < 2:
            return None
        try:
            return statistics.correlation(human, judge)
        except statistics.StatisticsError:
            return None


def validate_judge(llm: LLM, grades: list[HumanGrade] | None = None) -> ValidationReport:
    """Re-judge every human-graded plan and compare."""
    grades = load_human_grades() if grades is None else grades

    if not grades:
        return ValidationReport(
            pairs=[],
            note="no human grades recorded — run `uv run python -m eval.grade`",
        )

    pairs: list[tuple[float, float]] = []
    dimension_errors: dict[str, list[float]] = {k: [] for k in RUBRIC}

    for grade in grades:
        result = judge_plan(grade.case_id, grade.request, grade.plan, llm)
        if result.error or not result.scores:
            continue
        pairs.append((grade.overall, result.overall))
        for dimension, judge_score in result.scores.items():
            if dimension in grade.scores:
                dimension_errors[dimension].append(abs(grade.scores[dimension] - judge_score))

    report = ValidationReport(
        pairs=pairs,
        per_dimension_mae={
            k: sum(v) / len(v) for k, v in dimension_errors.items() if v
        },
    )
    if not report.usable:
        report.note = (
            f"only {report.n} graded plan(s); "
            f"{MIN_GRADES_FOR_VALIDATION} is the minimum for a meaningful number"
        )
    return report
