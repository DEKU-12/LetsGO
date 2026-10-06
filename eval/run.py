"""The evaluation report.

    uv run python -m eval.run                 # routing + a few judged plans
    uv run python -m eval.run --provider mock # free, no API key, structural only
    uv run python -m eval.run --plans 0       # routing metrics only, cheapest real run

Results are written to `eval_results/<timestamp>.json` so runs can be compared.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from backend.config import PROJECT_ROOT
from backend.graph import plan_trip
from backend.llm import LLM
from eval.dataset import ROUTING_CASES
from eval.judge import RUBRIC, judge_plan, validate_judge
from eval.selection import evaluate_selection
from eval.trajectory import check_trajectory

RESULTS_DIR = PROJECT_ROOT / "eval_results"

#: Cases used for plan generation and judging. Chosen to span the range: a full
#: trip, a partial one, and a single-agent question.
PLAN_CASE_IDS = ("full-japan", "trip-no-flights-porto", "itinerary-kyoto",
                 "stay-and-see-athens", "sights-lisbon")

BAR = "─" * 72


def _pct(value: float) -> str:
    return f"{value * 100:5.1f}%"


def _bar(value: float, width: int = 20) -> str:
    filled = round(value * width)
    return "█" * filled + "·" * (width - filled)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the evaluation suite.")
    parser.add_argument("--provider", default=None,
                        help="anthropic | groq | mock (default: auto from environment)")
    parser.add_argument("--plans", type=int, default=len(PLAN_CASE_IDS),
                        help="how many plans to generate and judge (0 = skip)")
    parser.add_argument("--no-save", action="store_true", help="do not write eval_results/")
    args = parser.parse_args()

    llm = LLM(provider=args.provider)
    started = datetime.now(timezone.utc)

    print(BAR)
    print(f"  LetsGO evaluation    {llm.provider} / {llm.model}")
    if llm.is_mock:
        print("  MOCK BACKEND — structural check only, these numbers judge nothing")
    print(BAR)

    # -- 1. agent selection -------------------------------------------------
    print("\n1. AGENT SELECTION" + f"   ({len(ROUTING_CASES)} labeled requests)\n")
    selection = evaluate_selection(llm)

    print(f"   {'agent':<22}{'prec':>7}{'recall':>8}{'F1':>7}   {'recall':<20}")
    for agent, score in selection.per_agent.items():
        print(f"   {agent:<22}{score.precision:>7.2f}{score.recall:>8.2f}"
              f"{score.f1:>7.2f}   {_bar(score.recall)}")

    micro = selection.micro
    print(f"\n   {'micro average':<22}{micro.precision:>7.2f}{micro.recall:>8.2f}{micro.f1:>7.2f}")
    print(f"   {'exact match':<22}{_pct(selection.exact_match):>22}"
          f"   {selection.exact_match * len(selection.results):.0f}"
          f"/{len(selection.results)} requests routed exactly right")
    print(f"   {'needed repair':<22}{_pct(selection.repaired):>22}"
          "   supervisor plans the code had to fix")

    wrong = [r for r in selection.results if not r.exact]
    if wrong:
        print("\n   misroutes:")
        for result in wrong:
            bits = []
            if result.missing:
                bits.append(f"missed {', '.join(sorted(result.missing))}")
            if result.extra:
                bits.append(f"added {', '.join(sorted(result.extra))}")
            print(f"     {result.case.id:<26} {'; '.join(bits) or 'clarified instead'}")

    # -- 2. plan quality ----------------------------------------------------
    judged, trajectories, grounding = [], [], []
    if args.plans > 0:
        chosen = [c for c in ROUTING_CASES if c.id in PLAN_CASE_IDS][: args.plans]
        print(f"\n\n2. PLAN QUALITY   ({len(chosen)} plans, judged 1-5 per dimension)\n")

        for case in chosen:
            state = plan_trip(case.request, llm)
            plan = state.get("final_plan") or ""
            result = judge_plan(case.id, case.request, plan, llm)
            judged.append(result)
            trajectories.append((case.id, check_trajectory(state)))

            research = state.get("research")
            checked = [a for a in (research.attractions if research else [])
                       if a.verified is not None]
            if checked:
                grounding.append((case.id, sum(a.verified for a in checked), len(checked),
                                  [a.name for a in checked if not a.verified]))

            if result.error:
                print(f"   {case.id:<26} judging failed: {result.error}")
                continue
            detail = "  ".join(f"{k[:4]} {v}" for k, v in result.scores.items())
            print(f"   {case.id:<26} {result.overall:>4.1f}/5   {detail}")

        scored = [r for r in judged if r.scores]
        if scored:
            mean = sum(r.overall for r in scored) / len(scored)
            print(f"\n   {'mean overall':<26} {mean:>4.1f}/5")
            for dimension in RUBRIC:
                values = [r.scores[dimension] for r in scored if dimension in r.scores]
                if values:
                    avg = sum(values) / len(values)
                    print(f"   {'  ' + dimension:<26} {avg:>4.1f}/5   {_bar(avg / 5)}")

        if grounding:
            total = sum(n for _, _, n, _ in grounding)
            confirmed = sum(c for _, c, _, _ in grounding)
            print(f"\n   {'places confirmed':<26} {confirmed}/{total} "
                  f"({confirmed / total:.0%}) against an independent map dataset")
            unconfirmed = [n for _, _, _, names in grounding for n in names]
            if unconfirmed:
                print("   could not confirm: " + ", ".join(unconfirmed[:6]))
                print("   (unconfirmed means not matched, not necessarily invented)")

    # -- 3. trajectory ------------------------------------------------------
    if trajectories:
        clean = sum(t.clean for _, t in trajectories)
        print(f"\n\n3. TRAJECTORY   ({clean}/{len(trajectories)} runs clean)\n")
        redone = sum(t.schedule_retries > 0 for _, t in trajectories)
        print(f"   schedule sent back for fixes in {redone}/{len(trajectories)} runs\n")
        for case_id, report in trajectories:
            if report.clean:
                print(f"   {case_id:<26} ok   {' -> '.join(
                    s for s in report.trace if not s.startswith('guard:'))}")
            else:
                print(f"   {case_id:<26} {'; '.join(report.issues)}")

    # -- 4. judge validation ------------------------------------------------
    print("\n\n4. JUDGE VALIDATION   (does the judge agree with a human?)\n")
    validation = validate_judge(llm)

    if not validation.pairs:
        print(f"   NOT VALIDATED — {validation.note}")
        print("   Until this is filled in, the scores in section 2 are unverified.")
    else:
        print(f"   graded plans           {validation.n}")
        print(f"   mean absolute error    {validation.mae:.2f} rubric points")
        print(f"   agree within 1 point   {_pct(validation.within_one)}")
        r = validation.correlation
        print(f"   correlation (Pearson)  {'n/a — too little spread' if r is None else f'{r:.2f}'}")
        if validation.per_dimension_mae:
            print("\n   error by dimension:")
            for dimension, mae in sorted(validation.per_dimension_mae.items(),
                                         key=lambda kv: -kv[1]):
                print(f"     {dimension:<16} {mae:.2f}   {_bar(1 - min(mae / 4, 1))}")
        if not validation.usable:
            print(f"\n   CAUTION — {validation.note}")

    elapsed = (datetime.now(timezone.utc) - started).total_seconds()
    print(f"\n{BAR}\n  {elapsed:.0f}s on {llm.provider}\n{BAR}")

    # -- persist ------------------------------------------------------------
    if not args.no_save:
        RESULTS_DIR.mkdir(exist_ok=True)
        path = RESULTS_DIR / f"{started:%Y%m%dT%H%M%SZ}.json"
        path.write_text(json.dumps({
            "timestamp": started.isoformat(),
            "provider": llm.provider,
            "model": llm.model,
            "selection": {
                "exact_match": selection.exact_match,
                "repaired": selection.repaired,
                "micro": {"precision": micro.precision, "recall": micro.recall, "f1": micro.f1},
                "per_agent": {
                    a: {"precision": s.precision, "recall": s.recall, "f1": s.f1,
                        "tp": s.tp, "fp": s.fp, "fn": s.fn}
                    for a, s in selection.per_agent.items()
                },
                "misroutes": [
                    {"case": r.case.id, "expected": sorted(r.case.expected),
                     "predicted": sorted(r.predicted), "reasoning": r.reasoning}
                    for r in selection.results if not r.exact
                ],
            },
            "plans": [{"case": r.case_id, "scores": r.scores, "overall": r.overall,
                       "not_applicable": r.not_applicable}
                      for r in judged],
            "grounding": [{"case": c, "confirmed": conf, "checked": n, "unconfirmed": names}
                          for c, conf, n, names in grounding],
            "trajectory": [{"case": c, "clean": t.clean, "issues": t.issues}
                           for c, t in trajectories],
            "judge_validation": {
                "n": validation.n, "mae": validation.mae,
                "within_one": validation.within_one,
                "correlation": validation.correlation, "note": validation.note,
            },
        }, indent=2))
        print(f"  saved {path.relative_to(PROJECT_ROOT)}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
