"""Destination Research agent — what the place is like.

Pulls weather through the weather adapter, asks the model for attractions and
practical notes, and writes a :class:`ResearchOutput` into shared state. Every
later agent reads from this, so it runs first whenever it runs at all.
"""

from __future__ import annotations

from typing import Any

from backend.adapters.places import PlacesAdapter
from backend.adapters.weather import WeatherAdapter
from backend.llm import LLM, LLMError, register_mock
from backend.state import Attraction, ResearchOutput, TravelState, WeatherOutlook

RESEARCH_SYSTEM = """You are a destination researcher. Given a place and a traveller's
interests, list what is genuinely worth their time.

Return JSON:
  {"attractions": [{"name": str, "category": str, "description": str, "est_hours": number}],
   "practical_notes": [str]}

Rules:
- 6 to 10 attractions, weighted toward the stated interests.
- "category" is one of: landmark, museum, food, nature, neighbourhood, experience.
- "est_hours" is realistic visit time including queueing.
- "practical_notes" covers visa, safety, money and transit basics — 3 to 5 short items.
- Only real places. If you are unsure a place exists, leave it out."""

# Small canned set so the mock backend produces something specific rather than
# generic filler. Clearly fake data for the no-API-key path.
_MOCK_ATTRACTIONS: dict[str, list[tuple[str, str, str, float]]] = {
    "japan": [
        ("Senso-ji Temple", "landmark", "Tokyo's oldest temple, busy but worth an early start.", 1.5),
        ("Tsukiji Outer Market", "food", "Street food stalls and knife shops.", 2.0),
        ("Fushimi Inari Shrine", "landmark", "Thousands of torii gates up the hillside.", 3.0),
        ("Nishiki Market", "food", "Covered market of Kyoto specialities.", 1.5),
        ("Tokyo National Museum", "museum", "The largest collection of Japanese art.", 2.5),
        ("Arashiyama Bamboo Grove", "nature", "Best walked before 8am.", 1.5),
        ("Shimokitazawa", "neighbourhood", "Vintage shops and small live venues.", 2.5),
        ("Kaiseki dinner", "experience", "Multi-course seasonal tasting menu.", 2.5),
    ],
    "italy": [
        ("Colosseum", "landmark", "Book a timed entry to skip the queue.", 2.0),
        ("Trastevere", "neighbourhood", "Evening food crawl territory.", 3.0),
        ("Vatican Museums", "museum", "Vast; pick two wings and stop.", 3.5),
        ("Testaccio Market", "food", "Where Romans actually shop.", 1.5),
        ("Pantheon", "landmark", "Free, ten minutes, extraordinary.", 1.0),
        ("Appian Way", "nature", "Cycle the old Roman road on a Sunday.", 3.0),
        ("Roman Forum", "landmark", "Pair with the Colosseum ticket.", 2.0),
    ],
    "france": [
        ("Musee d'Orsay", "museum", "Impressionists in a converted station.", 2.5),
        ("Le Marais", "neighbourhood", "Falafel, galleries, and old streets.", 3.0),
        ("Sainte-Chapelle", "landmark", "Go on a bright day for the glass.", 1.0),
        ("Marche d'Aligre", "food", "Market plus a standing wine bar.", 1.5),
        ("Canal Saint-Martin", "nature", "Evening picnic spot.", 2.0),
        ("Louvre", "museum", "Enter via Porte des Lions.", 3.0),
    ],
}

_MOCK_GENERIC = [
    ("Old town walking loop", "neighbourhood", "Orient yourself on foot the first morning.", 2.0),
    ("Central market", "food", "Local produce and cheap lunches.", 1.5),
    ("City history museum", "museum", "Context for everything else you will see.", 2.0),
    ("Viewpoint hike", "nature", "Best light in the late afternoon.", 2.5),
    ("Signature landmark", "landmark", "Book ahead in high season.", 1.5),
    ("Neighbourhood food crawl", "experience", "Three stops, one street.", 2.5),
]


_CITY_HINTS = {
    "japan": ("tokyo", "kyoto", "osaka"),
    "italy": ("rome", "florence", "venice", "milan"),
    "france": ("paris", "nice", "lyon"),
}


@register_mock("destination_research")
def _mock_research(context: dict[str, Any]) -> dict[str, Any]:
    destination = str(context.get("destination", "")).lower()
    rows = next(
        (v for k, v in _MOCK_ATTRACTIONS.items() if k in destination),
        None,
    )
    if rows is None:
        rows = next(
            (v for k, v in _MOCK_ATTRACTIONS.items() if any(c in destination for c in _CITY_HINTS.get(k, ()))),
            _MOCK_GENERIC,
        )
    return {
        "attractions": [
            {"name": n, "category": c, "description": d, "est_hours": h} for n, c, d, h in rows
        ],
        "practical_notes": [
            "Check visa requirements for your passport before booking.",
            "Card acceptance is common but carry some cash for markets.",
            "Buy a local transit pass on arrival rather than single tickets.",
            "Standard travel-safety precautions apply in crowded tourist areas.",
        ],
    }



def destination_research(state: TravelState, llm: LLM) -> dict[str, Any]:
    """Graph node: gather attractions, weather and practical notes."""
    params = state["params"]
    assert params is not None  # the guard node runs first

    weather_result = WeatherAdapter().fetch(destination=params.destination)
    sources = [f"{weather_result.provider}:{weather_result.source}"]

    try:
        raw = llm.json(
            task="destination_research",
            system=RESEARCH_SYSTEM,
            prompt=(
                f"Destination: {params.destination}\n"
                f"Trip length: {params.duration_days or 'unspecified'} days\n"
                f"Interests: {', '.join(params.preferences) or 'none stated'}\n"
                f"Travellers: {params.travelers}"
            ),
            context={"destination": params.destination, "preferences": params.preferences},
            max_tokens=2048,
        )
        errors: list[str] = []
    except LLMError as exc:
        raw = {"attractions": [], "practical_notes": []}
        errors = [f"destination_research: {exc}"]

    attractions = []
    for item in raw.get("attractions") or []:
        try:
            attractions.append(Attraction(**item))
        except (TypeError, ValueError):
            continue

    # Confirm the proposed places actually exist at this destination. The model
    # is good at choosing attractions and occasionally confident about ones that
    # are not there; this is the check on that.
    # An "experience" is an activity, not a location — a food crawl or a day
    # trip has no single point on a map, so verifying it would only manufacture
    # failures. Only physical places are checked.
    locatable = [a for a in attractions if a.category != "experience"]
    if locatable:
        places = PlacesAdapter().fetch(
            destination=params.destination,
            names=[a.name for a in locatable],
        )
        confirmed = places.data["confirmed"]
        if not places.is_mock:
            sources.append(f"{places.provider}:{places.source}")
            for attraction in locatable:
                attraction.verified = confirmed.get(attraction.name) is not None

    research = ResearchOutput(
        destination=params.destination,
        attractions=attractions,
        weather=WeatherOutlook(**weather_result.data),
        practical_notes=[str(n) for n in raw.get("practical_notes") or []],
        sources=sources + [f"llm:{llm.provider}"],
    )

    return {
        "research": research,
        "sources": sources,
        "trace": ["destination_research"],
        "errors": errors,
    }
