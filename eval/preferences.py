"""Does preference extraction keep what lasts and leave out what does not?

    uv run python -m eval.preferences
    uv run python -m eval.preferences --provider mock   # free, structural only

Each request is labelled with the lasting preferences it states and the
trip-only details it contains. Two numbers:

* **lasting caught** — each labelled lasting preference was suggested.
* **trip details leaked** — something about this trip only (an occasion, a
  budget "this time", a purpose) was suggested as worth remembering. This is
  the one that matters more: a leak means offering to store personal context
  the traveller never meant as a standing preference.

Matching is by keyword, since the model words things its own way ("vegetarian"
vs "no meat"); each label lists acceptable keywords. Results go to
`eval_results/preferences-<timestamp>.json`.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone

from backend.config import PROJECT_ROOT
from backend.llm import LLM, LLMError
from backend.preferences import suggest_preferences

RESULTS_DIR = PROJECT_ROOT / "eval_results"


@dataclass(frozen=True)
class PrefCase:
    id: str
    request: str
    #: One entry per lasting preference; any keyword in it counts as a match.
    lasting: tuple[tuple[str, ...], ...]
    #: Keywords that must not appear in any suggestion.
    trip_only: tuple[str, ...]


CASES: tuple[PrefCase, ...] = (
    PrefCase("diet-plus-occasion", "I'm vegetarian. 5 days in Rome for my mum's birthday",
             (("vegetarian", "meat"),), ("birthday", "mum", "mother")),
    PrefCase("kids-and-mornings", "4 days in Kyoto with our two kids, we hate early mornings",
             (("kid", "child"), ("morning", "early")), ()),
    PrefCase("budget-this-time", "Weekend in Paris, this time on a tight budget",
             (), ("budget", "cheap", "tight")),
    PrefCase("wheelchair", "My wife uses a wheelchair. 3 days in Lisbon, love food",
             (("wheelchair", "accessib", "mobility"),), ()),
    PrefCase("honeymoon", "Honeymoon in Greece, 7 days, somewhere romantic",
             (), ("honeymoon", "romantic")),
    PrefCase("allergy-and-alcohol", "I'm gluten-free and don't drink. 4 days in Barcelona",
             (("gluten", "coeliac", "celiac"), ("drink", "alcohol")), ()),
    PrefCase("business", "2 days in London for a conference, need to be near ExCeL",
             (), ("conference", "business", "excel")),
    PrefCase("slow-pace", "We always travel slowly, two sights a day at most. 5 days in Porto",
             (("slow", "two sights", "2 sights", "pace"),), ()),
    PrefCase("nothing-lasting", "5 days in Japan, mid-range, love food and history",
             (), ("japan", "mid-range")),
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate preference extraction.")
    parser.add_argument("--provider", default=None)
    args = parser.parse_args()

    llm = LLM(provider=args.provider)
    started = datetime.now(timezone.utc)
    print(f"preferences — {llm.provider} / {llm.model}"
          + ("   (MOCK: structural check only)" if llm.is_mock else ""))
    print(f"\n  {'case':<22}{'lasting':<10}{'leaked':<9}suggested")

    rows = []
    for case in CASES:
        try:
            found = suggest_preferences(case.request, llm, known=[])
        except LLMError as exc:
            print(f"  {case.id:<22}FAILED: {str(exc)[:90]}")
            rows.append({"id": case.id, "failed": str(exc)})
            continue

        text = " | ".join(found)
        caught = [group for group in case.lasting if any(k in text for k in group)]
        leaked = [k for k in case.trip_only if k in text]
        lasting = f"{len(caught)}/{len(case.lasting)}" if case.lasting else "-"
        print(f"  {case.id:<22}{lasting:<10}{', '.join(leaked) or '-':<9}{found}")
        rows.append({"id": case.id, "suggested": found, "caught": len(caught),
                     "expected": len(case.lasting), "leaked": leaked})

    scored = [r for r in rows if "failed" not in r]
    print(f"\n  lasting caught        {sum(r['caught'] for r in scored)}"
          f"/{sum(r['expected'] for r in scored)}")
    print(f"  trip details leaked   {sum(bool(r['leaked']) for r in scored)}/{len(scored)} requests")
    if len(scored) < len(rows):
        print(f"\n  WARNING — {len(rows) - len(scored)} case(s) failed on the model provider "
              "and are not counted above.")

    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"preferences-{started:%Y%m%dT%H%M%SZ}.json"
    path.write_text(json.dumps({"provider": llm.provider, "model": llm.model, "cases": rows},
                               indent=2))
    print(f"\n  saved {path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
