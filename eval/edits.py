"""How well does editing a plan by chat work?

    uv run python -m eval.edits
    uv run python -m eval.edits --provider mock   # free, structural only

Plans two base trips, then applies ten labelled change requests to them and
reports three things:

* **routing** — did the edit go to the right section, and the right days?
  Requests that would change the whole trip should be declined, not applied.
* **applied** — did the targeted part of the plan actually change? A correctly
  routed edit that returns the same schedule did nothing.
* **untouched** — did every day the traveller did not mention stay exactly as
  it was? `edit.py` enforces this in code, so anything below 100% is a bug.

Results go to `eval_results/edits-<timestamp>.json`.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone

from backend.config import PROJECT_ROOT
from backend.edit import NONE, edit_trip, route_edit
from backend.graph import plan_trip
from backend.llm import LLM

RESULTS_DIR = PROJECT_ROOT / "eval_results"

BASES = {
    "japan": "5 days in Japan, mid-range budget, love food and history, need hotels and flights",
    "rome": "4 days in Rome, love art and long lunches, need a hotel",
}


@dataclass(frozen=True)
class EditCase:
    id: str
    base: str
    message: str
    agent: str
    days: tuple[int, ...] = ()


CASES: tuple[EditCase, ...] = (
    EditCase("less-walking-day3", "japan", "less walking on day 3, my knees are bad", "itinerary", (3,)),
    EditCase("swap-days", "japan", "swap days 1 and 2", "itinerary", (1, 2)),
    EditCase("food-day4", "japan", "add more food stops on day 4", "itinerary", (4,)),
    EditCase("easy-last-day", "japan", "keep the last day light, we fly home that evening",
             "itinerary", (5,)),
    EditCase("cheaper-stay", "japan", "I'd rather stay somewhere cheaper", "accommodation"),
    EditCase("no-taxis", "japan", "skip taxis, we only want public transport", "transport"),
    EditCase("new-country", "japan", "actually let's do Korea instead", NONE),
    EditCase("late-starts", "rome", "nothing before 10am, I hate early mornings", "itinerary"),
    EditCase("near-vatican", "rome", "a hotel closer to the Vatican", "accommodation"),
    EditCase("question", "rome", "what's the weather going to be like?", NONE),
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate plan editing.")
    parser.add_argument("--provider", default=None)
    args = parser.parse_args()

    llm = LLM(provider=args.provider)
    started = datetime.now(timezone.utc)
    print(f"plan edits — {llm.provider} / {llm.model}"
          + ("   (MOCK: structural check only)" if llm.is_mock else ""))

    bases = {}
    for name, request in BASES.items():
        bases[name] = plan_trip(request, llm)
        failed = bases[name].get("errors") if not bases[name].get("final_plan") else None
        print(f"  base {name:<6} {'FAILED: ' + failed[0][:120] if failed else 'planned'}")

    rows = []
    print(f"\n  {'case':<20}{'routed':<9}{'applied':<9}{'untouched':<11}detail")
    for case in CASES:
        base = bases[case.base]
        decision = route_edit(base, case.message, llm)
        routed = decision["agent"] == case.agent and tuple(decision["days"]) == case.days
        detail = f"→ {decision['agent']} {decision['days'] or ''}".strip()

        applied = untouched = None
        failed = None
        if decision.get("error"):
            failed = decision["error"]
        elif decision["agent"] != NONE:
            edited, reply = edit_trip(base, case.message, llm)
            section = decision["agent"]
            if edited is None:
                failed = reply
            elif section == "itinerary":
                before, after = base["itinerary"].days, edited["itinerary"].days
                targets = set(decision["days"]) or {d.day for d in before}
                applied = any(a != b for a, b in zip(before, after) if b.day in targets)
                untouched = all(a == b for a, b in zip(before, after) if b.day not in targets)
            else:
                applied = edited.get(section) != base.get(section)

        if failed:
            # A model failure is an outage, not an answer: keep it out of the
            # routing and applied counts rather than scoring it as a mistake.
            routed = None
            detail = f"FAILED: {failed[:90]}"
        mark = lambda v: "-" if v is None else ("yes" if v else "NO")  # noqa: E731
        print(f"  {case.id:<20}{mark(routed):<9}{mark(applied):<9}{mark(untouched):<11}{detail}")
        rows.append({"id": case.id, "routed": routed, "applied": applied,
                     "untouched": untouched, "failed": failed, "decision": decision})

    def share(key: str) -> str:
        values = [r[key] for r in rows if r[key] is not None]
        return f"{sum(values)}/{len(values)}" if values else "n/a"

    print(f"\n  routed correctly        {share('routed')}")
    print(f"  change actually made    {share('applied')}")
    print(f"  other days untouched    {share('untouched')}")
    failures = sum(1 for r in rows if r["failed"])
    if failures:
        print(f"\n  WARNING — {failures} case(s) failed on the model provider and are not "
              "counted above. Rerun when the provider is available.")

    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"edits-{started:%Y%m%dT%H%M%SZ}.json"
    path.write_text(json.dumps({"provider": llm.provider, "model": llm.model,
                                "cases": rows}, indent=2, default=str))
    print(f"\n  saved {path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
