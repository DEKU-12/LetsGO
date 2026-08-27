"""Run one travel request end to end and print the result.

    uv run python scripts/run_once.py "5 days in Japan, mid-range, food and history"
"""

from __future__ import annotations

import sys

from backend.graph import plan_trip
from backend.llm import LLM


def main() -> int:
    request = " ".join(sys.argv[1:]) or (
        "5 days in Japan, mid-range budget, love food and history"
    )

    llm = LLM()
    print(f"request  : {request}")
    print(f"model    : {llm.provider} / {llm.model}")
    if llm.is_mock:
        print("           (no API key found — running on deterministic mock responses)")
    print()

    state = plan_trip(request, llm)

    routing = (state.get("meta") or {}).get("routing", {})
    print(f"route    : {' -> '.join(state.get('route_plan') or []) or '(none)'}")
    print(f"reason   : {routing.get('reasoning', '—')}")
    print(f"trace    : {' -> '.join(state.get('trace') or [])}")
    for err in state.get("errors") or []:
        print(f"note     : {err}")
    print()

    if state.get("clarification"):
        print(f"Needs clarification: {state['clarification']}")
        return 0

    print("-" * 72)
    print(state.get("final_plan") or "(no plan produced)")
    print("-" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
