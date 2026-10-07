"""Does the research agent call the right tools, with the right inputs?

    uv run python -m eval.tools
    uv run python -m eval.tools --provider mock   # free, structural only
    uv run python -m eval.tools --repeat 3        # pass^3: right on every try

Runs the input guard and the research agent (no other agents) on labelled
requests, and scores the tool calls the model chose to make:

* **right tool, right input** — for each city the trip covers, did it call
  ``get_weather`` for that city? Labelled per request: "5 days in Japan
  covering Tokyo and Kyoto" needs both.
* **invalid calls** — unknown tool, or arguments the tool rejected.
* **duplicate calls** — the same tool with the same arguments twice.
* **fallbacks** — the model never checked the weather, so code had to.

Results go to `eval_results/tools-<timestamp>.json`.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone

from backend.agents.destination_research import destination_research
from backend.config import PROJECT_ROOT
from backend.graph import apply_update
from backend.guards import parse_request
from backend.llm import LLM
from backend.state import new_state

RESULTS_DIR = PROJECT_ROOT / "eval_results"


@dataclass(frozen=True)
class ToolCase:
    id: str
    request: str
    #: Cities get_weather must be called for (matched case-insensitively,
    #: as a substring of the city argument).
    weather_cities: tuple[str, ...]


CASES: tuple[ToolCase, ...] = (
    ToolCase("single-city", "4 days in Lisbon, love food and history", ("lisbon",)),
    ToolCase("weather-question", "What's the weather like in Rome in October?", ("rome",)),
    ToolCase("two-cities", "5 days in Japan covering Tokyo and Kyoto, love food", ("tokyo", "kyoto")),
    ToolCase("three-cities", "A week in Italy: Rome, Florence and Venice, love art",
             ("rome", "florence", "venice")),
    ToolCase("peru", "10 days in Peru, Lima then Cusco for Machu Picchu", ("lima", "cusco")),
    ToolCase("weekend", "Weekend in Barcelona, love architecture", ("barcelona",)),
    ToolCase("country-only", "6 days in Portugal, beaches and wine", ()),
)


def _research(request: str, llm: LLM):
    state = new_state(request)
    apply_update(state, parse_request(state, llm))
    if state.get("clarification"):
        return state, (state.get("errors") or ["asked for clarification"])[0]
    apply_update(state, destination_research(state, llm))
    failed = [e for e in state.get("errors") or [] if e.startswith("destination_research:")]
    return state, failed[0] if failed else None


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the research agent's tool calls.")
    parser.add_argument("--provider", default=None)
    parser.add_argument("--repeat", type=int, default=1,
                        help="run every request N times and report pass^N")
    args = parser.parse_args()

    llm = LLM(provider=args.provider)
    started = datetime.now(timezone.utc)
    print(f"tool calls — {llm.provider} / {llm.model}"
          + ("   (MOCK: structural check only)" if llm.is_mock else ""))
    print(f"\n  {'case':<18}{'weather':<10}{'calls':<7}{'invalid':<9}{'dupes':<7}detail")

    rows = []
    passes: dict[str, list[bool]] = {}
    for run in range(args.repeat):
        for case in CASES:
            state, failure = _research(case.request, llm)
            if failure:
                passes.setdefault(case.id, []).append(False)
                if run == 0:
                    print(f"  {case.id:<18}FAILED: {failure[:100]}")
                    rows.append({"id": case.id, "failed": failure})
                continue

            calls = (state["meta"].get("tool_calls") or {}).get("destination_research", [])
            made = [c for c in calls if not c.get("fallback")]
            weather_args = [str((c.get("args") or {}).get("city", "")).lower()
                            for c in made if c["name"] == "get_weather" and c["ok"]]
            hit = [city for city in case.weather_cities if any(city in a for a in weather_args)]

            invalid = sum(not c["ok"] for c in made)
            keys = [(c["name"], json.dumps(c.get("args"), sort_keys=True).lower()) for c in made]
            dupes = len(keys) - len(set(keys))
            fallback = any(c.get("fallback") for c in calls)

            passes.setdefault(case.id, []).append(
                len(hit) == len(case.weather_cities) and not (invalid or dupes)
            )
            if run:  # repeats only feed pass^k
                continue

            detail = ", ".join(f"{c['name']}({', '.join(map(str, (c.get('args') or {}).values()))})"
                               for c in made) or "no calls"
            if fallback:
                detail += "  [weather fallback]"
            score = f"{len(hit)}/{len(case.weather_cities)}" if case.weather_cities else "-"
            print(f"  {case.id:<18}{score:<10}{len(made):<7}{invalid:<9}{dupes:<7}{detail[:110]}")

            rows.append({
                "id": case.id, "required": list(case.weather_cities), "hit": hit,
                "calls": len(made), "invalid": invalid, "duplicates": dupes,
                "fallback": fallback,
                "log": [{k: c.get(k) for k in ("name", "args", "ok", "error")} for c in calls],
            })

    scored = [r for r in rows if "failed" not in r]
    required = sum(len(r["required"]) for r in scored)
    print(f"\n  right tool, right city   {sum(len(r['hit']) for r in scored)}/{required}")
    print(f"  invalid calls            {sum(r['invalid'] for r in scored)} "
          f"of {sum(r['calls'] for r in scored)}")
    print(f"  duplicate calls          {sum(r['duplicates'] for r in scored)}")
    print(f"  weather fallbacks        {sum(r['fallback'] for r in scored)}/{len(scored)} runs")
    if args.repeat > 1:
        always = [cid for cid, runs in passes.items() if all(runs)]
        print(f"  pass^{args.repeat:<20}{len(always)}/{len(passes)} requests right on all "
              f"{args.repeat} runs (right cities, no invalid or duplicate calls)")
        shaky = [cid for cid in passes if cid not in always]
        if shaky:
            print(f"  not right on every run: {', '.join(shaky)}")
    total = llm.cost()
    if total is not None:
        print(f"  cost of this run         ${total:.2f}")
    if len(scored) < len(rows):
        print(f"\n  WARNING — {len(rows) - len(scored)} case(s) failed on the model provider "
              "and are not counted above.")

    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"tools-{started:%Y%m%dT%H%M%SZ}.json"
    path.write_text(json.dumps({"provider": llm.provider, "model": llm.model, "cases": rows},
                               indent=2, default=str))
    print(f"\n  saved {path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
