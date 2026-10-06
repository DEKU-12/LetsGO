"""Accommodation agent — which neighbourhoods to stay in.

Advice, not listings. Real hotel inventory and prices need a commercial
agreement, and a plan that shows invented hotels with invented prices teaches
the reader to distrust the rest of it. So this agent recommends *areas* — the
part of "where to stay" a well-read friend could actually tell you — and leaves
choosing a property to whatever booking site the traveller already uses.

Area names come from the model, so they are checked on a map the same way the
research agent's attractions are: "unverified" means "could not confirm", not
"does not exist".
"""

from __future__ import annotations

from typing import Any

from backend.adapters.places import PlacesAdapter
from backend.llm import LLM, LLMError, register_mock
from backend.state import AccommodationOutput, StayArea, TravelState, edit_note, profile_note

ACCOMMODATION_SYSTEM = """You advise a traveller on which neighbourhoods to stay in.

Return JSON:
  {"areas": [{"name": str, "why": str, "price_level": "budget" | "mid-range" | "luxury"}],
   "tips": [str]}

Rules:
- 2 or 3 real neighbourhoods or districts at the destination. Use the name as it
  appears on a map. If the trip covers several cities, cover the main ones.
- "why" is one sentence tying the area to this traveller: their interests, the
  places in their schedule, their budget.
- "price_level" is the typical level of places to stay in that area.
- Do not name hotels and do not state prices; you have no live inventory.
- "tips" is 1 to 3 short, practical booking tips for this destination (when to
  book, what to look for). No prices."""

_MOCK_AREAS: dict[str, list[tuple[str, str, str]]] = {
    "japan": [
        ("Asakusa", "Traditional Tokyo, walking distance to Senso-ji.", "mid-range"),
        ("Shinjuku", "Transport hub with late-night food everywhere.", "mid-range"),
        ("Gion", "Kyoto's old quarter, close to the temples.", "luxury"),
    ],
    "italy": [
        ("Trastevere", "Evening food scene on the doorstep.", "mid-range"),
        ("Monti", "Central, a short walk to the Colosseum.", "mid-range"),
    ],
    "france": [
        ("Le Marais", "Central, walkable, full of galleries and food.", "mid-range"),
        ("Saint-Germain-des-Prés", "Classic Left Bank, near the museums.", "luxury"),
    ],
}

_MOCK_GENERIC = [
    ("Old town", "Walkable and close to most sights.", "mid-range"),
    ("Station district", "Cheaper and well connected.", "budget"),
]


@register_mock("accommodation")
def _mock_accommodation(context: dict[str, Any]) -> dict[str, Any]:
    destination = str(context.get("destination", "")).lower()
    rows = next((v for k, v in _MOCK_AREAS.items() if k in destination), _MOCK_GENERIC)
    return {
        "areas": [{"name": n, "why": w, "price_level": p} for n, w, p in rows],
        "tips": ["Book refundable rates until your dates are fixed."],
    }


def accommodation(state: TravelState, llm: LLM) -> dict[str, Any]:
    """Graph node: recommend neighbourhoods to stay in."""
    params = state["params"]
    assert params is not None

    # Staying near what you will actually do is most of the advice, so give
    # the model the schedule when there is one, else the researched places.
    itinerary = state.get("itinerary")
    research = state.get("research")
    if itinerary and itinerary.days:
        anchors = [b.title for d in itinerary.days for b in d.blocks]
    else:
        anchors = [a.name for a in research.attractions] if research else []

    try:
        raw = llm.json(
            task="accommodation",
            system=ACCOMMODATION_SYSTEM,
            prompt=(
                f"Destination: {params.destination}\n"
                f"Budget level: {params.budget or 'unstated'}\n"
                f"Nights: {max((params.duration_days or 1) - 1, 1)}   "
                f"Travellers: {params.travelers}\n"
                f"Interests: {', '.join(params.preferences) or 'none stated'}\n"
                f"Places they plan to visit: {', '.join(anchors[:20]) or 'not planned yet'}"
                f"{profile_note(state)}"
                f"{edit_note(state)}"
            ),
            context={"destination": params.destination},
            max_tokens=1024,
        )
        errors: list[str] = []
    except LLMError as exc:
        raw, errors = {}, [f"accommodation: {exc}"]

    areas: list[StayArea] = []
    for item in raw.get("areas") or []:
        try:
            areas.append(StayArea(**item))
        except (TypeError, ValueError):
            continue

    sources: list[str] = []
    if areas:
        places = PlacesAdapter().fetch(destination=params.destination, names=[a.name for a in areas])
        if not places.is_mock:
            sources.append(f"{places.provider}:{places.source}")
            for area in areas:
                area.verified = places.data["confirmed"].get(area.name) is not None

    return {
        "accommodation": AccommodationOutput(
            areas=areas, tips=[str(t) for t in raw.get("tips") or []]
        ),
        "sources": sources,
        "errors": errors,
        "trace": ["accommodation"],
    }
