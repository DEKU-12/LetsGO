"""The evaluation report.

    uv run python -m eval.run                 # routing + a few judged plans
    uv run python -m eval.run --provider mock # free, no API key, structural only
    uv run python -m eval.run --plans 0       # routing metrics only, cheapest real run
    uv run python -m eval.run --plans 0 --repeat 3   # routing pass^3: right on every try

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
    parser.add_argument("--repeat", type=int, default=1,
                        help="route every request N times and report pass^N (right on all N)")
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

    failed = [r for r in selection.results if r.error]
    if failed:
        print(f"\n   WARNING — {len(failed)}/{len(selection.results)} requests never got a "
              "routing decision, so the numbers above are not a measurement.")
        print(f"   first failure: {failed[0].error[:200]}")

    # pass^k: the model is not deterministic, so one correct run per request is
    # pass@1. A request counts toward pass^k only if every one of k runs routed
    # it exactly right — the number that says whether it can be relied on.
    passk = None
    if args.repeat > 1:
        runs = [selection, *(evaluate_selection(llm) for _ in range(args.repeat - 1))]
        always = [all(run.results[i].exact for run in runs) for i in range(len(selection.results))]
        passk = sum(always) / len(always)
        print(f"   {f'pass^{args.repeat}':<22}{_pct(passk):>22}"
              f"   {sum(always)}/{len(always)} routed exactly right on all {args.repeat} runs")
        flaky = [r.case.id for r, ok in zip(selection.results, always) if not ok]
        if flaky:
            print(f"   not right on every run: {', '.join(flaky)}")

    wrong = [r for r in selection.results if not r.exact and not r.error]
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

        spent: list[dict[str, list[int]]] = []
        seconds: list[float] = []
        for case in chosen:
            before = {k: list(v) for k, v in llm.usage.items()}
            clock = datetime.now(timezone.utc)
            state = plan_trip(case.request, llm)
            seconds.append((datetime.now(timezone.utc) - clock).total_seconds())
            spent.append({k: [a - b for a, b in zip(v, before.get(k, [0, 0, 0]))]
                          for k, v in llm.usage.items()})
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
                reason = result.error
                if not plan and state.get("errors"):
                    reason = f"no plan — {state['errors'][0][:160]}"
                print(f"   {case.id:<26} judging failed: {reason}")
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

        if spent:
            # Cost and size per plan (generation only, not judging), and where
            # the tokens go — the per-agent split is what to optimise first.
            per_plan = [sum(i + o for _, i, o in s.values()) for s in spent]
            by_task: dict[str, list[int]] = {}
            for s in spent:
                for task, (c, i, o) in s.items():
                    row = by_task.setdefault(task, [0, 0, 0])
                    row[0] += c; row[1] += i; row[2] += o  # noqa: E702
            costs = [llm.cost(s) for s in spent]
            print(f"\n   {'seconds per plan':<26} {sum(seconds) / len(seconds):.0f} average, "
                  f"{max(seconds):.0f} slowest")
            print(f"   {'tokens per plan':<26} {sum(per_plan) // len(per_plan):,} average"
                  + (f", ${sum(costs) / len(costs):.3f} average" if None not in costs else ""))
            for task, (c, i, o) in sorted(by_task.items(), key=lambda kv: -(kv[1][1] + kv[1][2])):
                if not c:
                    continue  # tasks seen in this run but not while generating plans
                print(f"     {task:<24} {c:>3} calls  {i + o:>8,} tokens")

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
            if report.unresearched_stops:
                print(f"   {'':<26} not a researched place: "
                      f"{', '.join(report.unresearched_stops[:4])}")

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
    total = llm.cost()
    print(f"\n{BAR}\n  {elapsed:.0f}s on {llm.provider}"
          + (f", ${total:.2f} for this whole run" if total is not None else "") + f"\n{BAR}")

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
                "repeat": args.repeat,
                "pass_k": passk,
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
