"""Schedule check — is the itinerary actually doable?

Runs straight after the itinerary agent. Every check here is plain code, not a
model call: a model grading its own homework tends to approve it, while
arithmetic on visit hours and map distances does not.

If anything fails, the itinerary agent gets the list of problems and one more
try, up to ``MAX_RETRIES`` times. After that the plan goes ahead and the
remaining problems are recorded in ``errors`` rather than hidden.

Checks:

* **missing days** — fewer days than the traveller asked for, or an empty one.
* **overloaded day** — the scheduled visits add up to more than a person can do.
* **repeated stop** — the same attraction booked on two days.
* **too spread out** — one day's stops are far apart on the map. Only runs for
  places the verifier located, so it is silent on the mock backend.

Opening hours are not checked: the geocoder used for verification does not
return them, and a per-place details lookup is a separate API call per stop.
"""

from __future__ import annotations

import math
from typing import Any

from backend.adapters.places import name_variants, names_match
from backend.agents.itinerary import DEFAULT_DAYS
from backend.state import Attraction, TravelState

#: How many times the itinerary agent may redo the schedule.
MAX_RETRIES = 2

#: Scheduled visit time per day before the day is unrealistic. Visit estimates
#: already include queueing; meals and travel come on top.
MAX_HOURS_PER_DAY = 9.0

#: Straight-line distance between one day's stops, summed in visiting order.
#: Generous enough for a day trip from the base city (Kyoto to Nara is ~35 km),
#: tight enough to catch Tokyo and Kyoto on the same day (~370 km).
MAX_KM_PER_DAY = 60.0


def match_attraction(title: str, attractions: list[Attraction]) -> Attraction | None:
    """Which researched attraction, if any, an itinerary block is a visit to."""
    lowered = title.lower()
    for attraction in attractions:
        variants = name_variants(attraction.name)
        if any(v.lower() in lowered for v in variants):
            return attraction
        if any(names_match(v, title) for v in variants):
            return attraction
    return None


def _km(a: Attraction, b: Attraction) -> float:
    """Great-circle distance in kilometres."""
    lat1, lon1, lat2, lon2 = map(math.radians, (a.lat, a.lon, b.lat, b.lon))
    h = (
        math.sin((lat2 - lat1) / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    )
    return 2 * 6371.0 * math.asin(math.sqrt(h))


def find_problems(state: TravelState) -> list[str]:
    """Everything wrong with the current itinerary, as instructions to fix it."""
    params = state.get("params")
    itinerary = state.get("itinerary")
    research = state.get("research")
    if params is None or itinerary is None or research is None:
        return []

    attractions = research.attractions
    days_wanted = params.duration_days or DEFAULT_DAYS
    problems: list[str] = []

    if len(itinerary.days) < days_wanted:
        problems.append(
            f"Only {len(itinerary.days)} of {days_wanted} days were planned. "
            f"Plan all {days_wanted} days."
        )

    first_seen: dict[str, int] = {}
    for day in itinerary.days:
        if not day.blocks:
            problems.append(f"Day {day.day} is empty. Give it at least one activity.")
            continue

        # A place can appear in two blocks ("Nishiki Market", "Lunch at Nishiki
        # Market"); it is still one visit.
        matched = (match_attraction(b.title, attractions) for b in day.blocks)
        visits = list({a.name: a for a in matched if a}.values())

        hours = sum(a.est_hours for a in visits)
        if hours > MAX_HOURS_PER_DAY:
            problems.append(
                f"Day {day.day} has {hours:g} hours of visits (limit {MAX_HOURS_PER_DAY:g}). "
                "Move or drop something."
            )

        for attraction in dict.fromkeys(a.name for a in visits):
            if attraction in first_seen and first_seen[attraction] != day.day:
                problems.append(
                    f"{attraction} is on day {first_seen[attraction]} and day {day.day}. "
                    "Keep it on one day."
                )
            first_seen.setdefault(attraction, day.day)

        located = [a for a in visits if a.lat is not None and a.lon is not None]
        km = sum(_km(a, b) for a, b in zip(located, located[1:]))
        if km > MAX_KM_PER_DAY:
            problems.append(
                f"Day {day.day}'s stops are about {km:.0f} km apart "
                f"({', '.join(a.name for a in located)}). Group nearby places on the same day."
            )

    return problems


def check_itinerary(state: TravelState) -> dict[str, Any]:
    """Graph node: check the schedule, and either send it back or let it through."""
    problems = find_problems(state)
    attempts = state.get("check_attempts", 0) + 1

    meta = dict(state.get("meta") or {})
    meta["itinerary_checks"] = [*meta.get("itinerary_checks", []), problems]

    update: dict[str, Any] = {
        "itinerary_feedback": problems,
        "check_attempts": attempts,
        "meta": meta,
        "trace": ["check_itinerary"],
    }
    if problems and attempts > MAX_RETRIES:
        update["errors"] = [
            f"check_itinerary: unresolved after {MAX_RETRIES} retries: {p}" for p in problems
        ]
    return update


def after_check(state: TravelState) -> str:
    """Redo the itinerary if the check failed and retries remain."""
    if state.get("itinerary_feedback") and state.get("check_attempts", 0) <= MAX_RETRIES:
        return "itinerary"
    return "supervisor"
