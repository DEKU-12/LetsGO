"""Does the supervisor call the right agents?

Reported three ways, because they fail differently:

* **Per-agent precision/recall** — which agents get over- or under-invoked.
  Recall matters most: a missing agent means a missing section of the plan.
  Precision matters for cost and clutter.
* **Micro-averaged precision/recall/F1** — one headline number over all
  agent decisions.
* **Exact match** — the fraction of requests where the routing set was exactly
  right. The strictest measure, and the one that is hardest to game.

Only the four workers are scored. The aggregator always runs, so counting it
would add a guaranteed true positive to every case.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from backend.agents.supervisor import CANONICAL_ORDER, supervisor
from backend.guards import parse_request
from backend.llm import LLM
from backend.state import AGGREGATOR, new_state
from eval.dataset import ROUTING_CASES, RoutingCase


@dataclass
class CaseResult:
    case: RoutingCase
    predicted: frozenset[str]
    clarified: bool
    repairs: list[str] = field(default_factory=list)
    reasoning: str = ""
    error: str | None = None

    @property
    def exact(self) -> bool:
        return self.predicted == self.case.expected

    @property
    def missing(self) -> frozenset[str]:
        return self.case.expected - self.predicted

    @property
    def extra(self) -> frozenset[str]:
        return self.predicted - self.case.expected


@dataclass
class AgentScore:
    agent: str
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 1.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 1.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0


@dataclass
class SelectionReport:
    results: list[CaseResult]
    per_agent: dict[str, AgentScore]

    @property
    def exact_match(self) -> float:
        return sum(r.exact for r in self.results) / len(self.results) if self.results else 0.0

    @property
    def micro(self) -> AgentScore:
        total = AgentScore("micro")
        for score in self.per_agent.values():
            total.tp += score.tp
            total.fp += score.fp
            total.fn += score.fn
        return total

    @property
    def repaired(self) -> float:
        """Share of runs where the code had to fix the supervisor's plan."""
        if not self.results:
            return 0.0
        return sum(bool(r.repairs) for r in self.results) / len(self.results)


def route_once(case: RoutingCase, llm: LLM) -> CaseResult:
    """Run the guard and the supervisor only — no workers, so this stays cheap."""
    state = new_state(case.request)

    try:
        state.update(parse_request(state, llm))
    except Exception as exc:  # noqa: BLE001 - a crash is a result, not a stop
        return CaseResult(case, frozenset(), False, error=f"parse: {exc}")

    if state.get("clarification"):
        # Asking for clarification is the correct answer to an unplannable
        # request: no agents invoked.
        return CaseResult(case, frozenset(), True)

    try:
        state.update(supervisor(state, llm, CANONICAL_ORDER))
    except Exception as exc:  # noqa: BLE001
        return CaseResult(case, frozenset(), False, error=f"supervisor: {exc}")

    plan = [a for a in state.get("route_plan", []) if a != AGGREGATOR]
    routing = (state.get("meta") or {}).get("routing", {})

    return CaseResult(
        case=case,
        predicted=frozenset(plan),
        clarified=False,
        repairs=list(routing.get("repairs") or []),
        reasoning=str(routing.get("reasoning", "")),
    )


def evaluate_selection(llm: LLM, cases=ROUTING_CASES) -> SelectionReport:
    results = [route_once(case, llm) for case in cases]

    per_agent = {agent: AgentScore(agent) for agent in CANONICAL_ORDER}
    for result in results:
        for agent in CANONICAL_ORDER:
            expected = agent in result.case.expected
            predicted = agent in result.predicted
            if expected and predicted:
                per_agent[agent].tp += 1
            elif predicted:
                per_agent[agent].fp += 1
            elif expected:
                per_agent[agent].fn += 1

    return SelectionReport(results=results, per_agent=per_agent)
