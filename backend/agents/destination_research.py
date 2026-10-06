"""Destination Research agent — what the place is like.

The one agent that uses tool calling: the model decides what to look up.

* ``get_weather(city)`` — the model picks the cities. A trip "covering Tokyo
  and Kyoto" gets weather for both, where a fixed call could only ask about
  "Japan".
* ``check_place(name, city)`` — the model can confirm a place it is unsure of
  before recommending it, and pick another if it cannot be found.

What the model decides is flexible; what must always happen is not left to it.
If it never checks the weather, the destination's weather is fetched anyway,
and every place it recommends is still verified against the map afterwards —
that check is the project's guarantee, so it stays in code. Every tool call is
logged in ``meta["tool_calls"]`` for ``eval/tools.py``.

Writes a :class:`ResearchOutput` into shared state. Every later agent reads
from this, so it runs first whenever it runs at all.
"""

from __future__ import annotations

from typing import Any

from backend.adapters.places import PlacesAdapter, name_variants, names_match
from backend.adapters.weather import WeatherAdapter
from backend.adapters.wikimedia import WikimediaAdapter
from backend.llm import LLM, LLMError, Tool, register_mock
from backend.state import (
    Attraction,
    Photo,
    ResearchOutput,
    TravelState,
    WeatherOutlook,
    profile_note,
)

RESEARCH_SYSTEM = """You are a destination researcher. Given a place and a traveller's
interests, list what is genuinely worth their time.

You have tools:
- get_weather(city): the weather outlook for one city. Call it for each main
  city the trip covers (at most 3). For a country, pick the city they will
  spend most time in.
- check_place(name, city): whether a place can be found on a map. Use it on
  places you are not sure of (at most 4 calls). If one comes back
  "exists": false, leave it out and choose another.

Return JSON:
  {"attractions": [{"name": str, "category": str, "description": str, "est_hours": number}],
   "practical_notes": [str]}

Rules:
- 6 to 10 attractions, weighted toward the stated interests.
- "category" is one of: landmark, museum, food, nature, neighbourhood, experience.
- "est_hours" is realistic visit time including queueing.
- "practical_notes" covers visa, safety, money and transit basics — 3 to 5 short items.
- Only real places. If you are unsure a place exists, leave it out.
- One place per entry, named as it appears on a map. Do not combine places
  ("South Beach and Ocean Drive"): name the main one, or list them separately."""

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


@register_mock("destination_research:tools")
def _mock_tool_calls(context: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"name": "get_weather", "args": {"city": context.get("destination", "")}}]


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



def _photos(details: dict[str, dict[str, Any]], sources: set[str]) -> dict[str, Any]:
    """Photos for every Wikidata id the map entries carried, in one batch."""
    qids = [d["wikidata"] for d in details.values() if d.get("wikidata")]
    if not qids:
        return {}
    result = WikimediaAdapter().fetch(qids=qids)
    if result.data:
        sources.add(f"{result.provider}:{result.source}")
    return result.data


def _photo_for(
    name: str, details: dict[str, dict[str, Any]], photos: dict[str, Any]
) -> Photo | None:
    photo = photos.get((details.get(name) or {}).get("wikidata") or "")
    return _photo(name, photo) if photo else None


def attach_photos(
    attractions: list[Attraction],
    destination: str,
    details: dict[str, dict[str, Any]],
    centre: dict[str, float] | None,
    sources: set[str],
) -> Photo | None:
    """Give every attraction a photo, and return the destination's.

    In order of trust:
    1. The photo linked from the map entry that verified the place.
    2. A Wikipedia search, accepted only when the article's title matches and
       its coordinates are near where the map put the place. Needs a known
       location, so unverified places skip it.
    3. A photo of the destination, labelled as such — never passed off as the
       place itself.
    """
    photos = _photos(details, sources)
    for attraction in attractions:
        attraction.photo = _photo_for(attraction.name, details, photos)
    destination_photo = _photo_for(destination, details, photos)

    searches = [
        {"key": a.name, "query": f"{a.name} {destination}", "names": name_variants(a.name),
         "lat": a.lat, "lon": a.lon}
        for a in attractions if a.photo is None and a.lat is not None and a.lon is not None
    ]
    if destination_photo is None and centre:
        searches.append({"key": destination, "query": destination,
                         "names": [destination], "lat": centre["lat"], "lon": centre["lon"]})
    if searches:
        result = WikimediaAdapter().fetch(searches=searches)
        if result.data:
            sources.add(f"wikipedia:{result.source}")
        found = {key: _photo(key, photo) for key, photo in result.data.items()}
        for attraction in attractions:
            attraction.photo = attraction.photo or found.get(attraction.name)
        destination_photo = destination_photo or found.get(destination)

    if destination_photo:
        stand_in = destination_photo.model_copy(update={"caption": destination, "generic": True})
        for attraction in attractions:
            attraction.photo = attraction.photo or stand_in
    return destination_photo


def _photo(name: str, photo: dict[str, Any]) -> Photo:
    # The linked item can be something related rather than the place itself
    # (a café to its famous pastry). Say what the photo shows.
    label = photo.get("label") or ""
    caption = None if not label or names_match(name, label) else label
    return Photo(url=photo["url"], page=photo["page"], credit=photo["credit"], caption=caption)


def destination_research(state: TravelState, llm: LLM) -> dict[str, Any]:
    """Graph node: gather attractions, weather and practical notes."""
    params = state["params"]
    assert params is not None  # the guard node runs first

    sources: set[str] = set()
    city_weather: dict[str, WeatherOutlook] = {}

    def get_weather(city: str) -> dict[str, Any]:
        result = WeatherAdapter().fetch(destination=city)
        sources.add(f"{result.provider}:{result.source}")
        city_weather[city] = WeatherOutlook(**result.data)
        return result.data

    def check_place(name: str, city: str) -> dict[str, Any]:
        result = PlacesAdapter().fetch(destination=city, names=[name])
        if result.is_mock:
            return {"exists": None, "note": "no map data available; use your judgement"}
        sources.add(f"{result.provider}:{result.source}")
        matched = result.data["confirmed"].get(name)
        return {"exists": matched is not None, "matched_name": matched}

    tools = [
        Tool("get_weather", "Weather outlook for one city over the next few days.",
             {"type": "object", "properties": {"city": {"type": "string"}},
              "required": ["city"]},
             get_weather),
        Tool("check_place", "Whether a named place can be found on a map near a city.",
             {"type": "object",
              "properties": {"name": {"type": "string"}, "city": {"type": "string"}},
              "required": ["name", "city"]},
             check_place),
    ]

    calls: list[dict[str, Any]] = []
    try:
        raw, calls = llm.run_tools(
            task="destination_research",
            system=RESEARCH_SYSTEM,
            prompt=(
                f"Destination: {params.destination}\n"
                f"Trip length: {params.duration_days or 'unspecified'} days\n"
                f"Interests: {', '.join(params.preferences) or 'none stated'}\n"
                f"Travellers: {params.travelers}\n"
                f"Request in their words: {state.get('request', '')}"
                f"{profile_note(state)}"
            ),
            tools=tools,
            context={"destination": params.destination, "preferences": params.preferences},
            max_tokens=4096,
        )
        errors: list[str] = []
    except LLMError as exc:
        raw = {"attractions": [], "practical_notes": []}
        errors = [f"destination_research: {exc}"]

    # The plan always gets a weather section: if the model never asked, ask.
    if not city_weather:
        get_weather(params.destination)
        calls.append({"name": "get_weather", "args": {"city": params.destination},
                      "ok": True, "fallback": True})

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
    details: dict[str, dict[str, Any]] = {}
    if locatable:
        places = PlacesAdapter().fetch(
            destination=params.destination,
            names=[a.name for a in locatable],
            details=True,
        )
        confirmed = places.data["confirmed"]
        if not places.is_mock:
            sources.add(f"{places.provider}:{places.source}")
            coords = places.data.get("coords") or {}
            details = places.data.get("details") or {}
            for attraction in locatable:
                attraction.verified = confirmed.get(attraction.name) is not None
                if attraction.name in coords:
                    attraction.lat, attraction.lon = coords[attraction.name]
                attraction.opening_hours = (details.get(attraction.name) or {}).get("opening_hours")

    centre = places.data.get("centre") if locatable and not places.is_mock else None
    destination_photo = attach_photos(attractions, params.destination, details, centre, sources)

    research = ResearchOutput(
        destination=params.destination,
        attractions=attractions,
        weather=next(iter(city_weather.values())),
        city_weather=city_weather,
        photo=destination_photo,
        practical_notes=[str(n) for n in raw.get("practical_notes") or []],
        sources=sorted(sources) + [f"llm:{llm.provider}"],
    )

    meta = dict(state.get("meta") or {})
    meta["tool_calls"] = {**meta.get("tool_calls", {}), "destination_research": calls}

    return {
        "research": research,
        "sources": sorted(sources),
        "meta": meta,
        "trace": ["destination_research"],
        "errors": errors,
    }
