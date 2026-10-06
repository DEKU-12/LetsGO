"""Transport agent — how the traveller gets there and gets around.

Practical guidance, not fares. Live fare data needs a commercial agreement, and
a made-up price stated confidently is worse than none, so this agent names the
usual way in (main airports, rail hubs), the sensible ways to get around, and
the passes or apps worth knowing about — and leaves prices to the booking site.
"""

from __future__ import annotations

from typing import Any

from backend.llm import LLM, LLMError, register_mock
from backend.state import TransportOutput, TransportTip, TravelState, edit_note, profile_note

TRANSPORT_SYSTEM = """You advise on travel to and around a destination.

Return JSON:
  {"arrival": [str], "local": [{"mode": str, "description": str}], "tips": [str]}

Rules:
- "arrival": 1 or 2 lines on how visitors usually arrive — the main international
  airport(s) or rail hubs, and how to get from there into town.
- "local": 2 to 4 ways to get around, each with one sentence suited to this trip's
  length, group size and the places in their schedule.
- "tips": 1 to 3 short practical tips — passes, transit cards, apps, what to skip.
- No prices, fares or timetables; you have no live data. Real place names only."""


@register_mock("transport")
def _mock_transport(context: dict[str, Any]) -> dict[str, Any]:
    destination = context.get("destination", "the city")
    return {
        "arrival": [f"Most visitors fly into {destination}'s main international airport "
                    "and take the rail link or a taxi into the centre."],
        "local": [
            {"mode": "Metro", "description": "Fastest across town; buy a stored-value card."},
            {"mode": "Walking", "description": "Most central sights are close together."},
        ],
        "tips": ["Buy any multi-day transit pass at the airport on arrival."],
    }


def transport(state: TravelState, llm: LLM) -> dict[str, Any]:
    """Graph node: advise how to get there and get around."""
    params = state["params"]
    assert params is not None

    itinerary = state.get("itinerary")
    stops = [b.title for d in itinerary.days for b in d.blocks] if itinerary else []

    try:
        raw = llm.json(
            task="transport",
            system=TRANSPORT_SYSTEM,
            prompt=(
                f"Destination: {params.destination}\n"
                f"Trip length: {params.duration_days or 'unspecified'} days\n"
                f"Travellers: {params.travelers}\n"
                f"Places in their schedule: {', '.join(stops[:20]) or 'not planned yet'}"
                f"{profile_note(state)}"
                f"{edit_note(state)}"
            ),
            context={"destination": params.destination},
            max_tokens=1024,
        )
        errors: list[str] = []
    except LLMError as exc:
        raw, errors = {}, [f"transport: {exc}"]

    local: list[TransportTip] = []
    for item in raw.get("local") or []:
        try:
            local.append(TransportTip(**item))
        except (TypeError, ValueError):
            continue

    return {
        "transport": TransportOutput(
            arrival=[str(a) for a in raw.get("arrival") or []],
            local=local,
            tips=[str(t) for t in raw.get("tips") or []],
        ),
        "errors": errors,
        "trace": ["transport"],
    }
