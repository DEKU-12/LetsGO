"""Was the path through the graph sane, or merely successful?

A run can produce a good-looking plan while doing something wrong on the way —
running an agent whose output nobody reads, or reaching a worker before its
prerequisite. Those are process failures, invisible in the output, and they are
what this module reports.

Four checks:

* **invalid_order** — a worker ran before something it depends on.
* **unused_output** — a worker ran but produced nothing, or produced something
  the final plan never shows. Wasted tokens and latency.
* **repaired_plan** — the supervisor's route needed fixing before it could run.
* **agent_errors** — anything an agent recorded about its own run.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from backend.agents.supervisor import CANONICAL_ORDER, DEPENDENCIES
from backend.state import ACCOMMODATION, ITINERARY, RESEARCH, TRANSPORT, TravelState

#: The state slot each worker fills, and the heading its output should reach.
_CONTRIBUTION: dict[str, tuple[str, str]] = {
    RESEARCH: ("research", "## The destination"),
    ITINERARY: ("itinerary", "## Day by day"),
    ACCOMMODATION: ("accommodation", "## Where to stay"),
    TRANSPORT: ("transport", "## Getting there and around"),
}


@dataclass
class TrajectoryReport:
    trace: list[str]
    invalid_order: list[str] = field(default_factory=list)
    unused_output: list[str] = field(default_factory=list)
    repaired_plan: list[str] = field(default_factory=list)
    agent_errors: list[str] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not (self.invalid_order or self.unused_output or self.repaired_plan)

    @property
    def issues(self) -> list[str]:
        return [
            *(f"out of order: {i}" for i in self.invalid_order),
            *(f"unused output: {i}" for i in self.unused_output),
            *(f"plan repaired: {i}" for i in self.repaired_plan),
        ]


def check_trajectory(state: TravelState) -> TrajectoryReport:
    """Inspect one finished run."""
    trace = list(state.get("trace") or [])
    workers = [step for step in trace if step in CANONICAL_ORDER]
    plan = state.get("final_plan") or ""

    report = TrajectoryReport(trace=trace)

    # 1. Dependencies must be satisfied before a worker runs.
    for position, worker in enumerate(workers):
        for need in DEPENDENCIES.get(worker, ()):
            if need not in workers[:position]:
                report.invalid_order.append(f"{worker} ran before {need}")

    # 2. Every worker that ran should have reached the finished plan.
    for worker in workers:
        slot, heading = _CONTRIBUTION[worker]
        produced = state.get(slot)
        if not produced:
            report.unused_output.append(f"{worker} produced nothing")
        elif plan and heading not in plan:
            report.unused_output.append(f"{worker} output missing from the plan")

    # 3. Did the supervisor's route need repairing?
    routing = (state.get("meta") or {}).get("routing", {})
    report.repaired_plan = list(routing.get("repairs") or [])

    # 4. Anything the agents flagged about themselves.
    report.agent_errors = list(state.get("errors") or [])

    return report
