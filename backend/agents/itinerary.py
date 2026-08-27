"""Itinerary Planning agent — turns research into a day-by-day schedule.

Reads what Destination Research found and arranges it across the trip. The
prompt constrains the model to the attractions already in state: the schedule
may add meals, rest and travel time, but it may not introduce places nobody
researched. That keeps the plan grounded in the same set the research agent
can be held to.
"""

from __future__ import annotations

from typing import Any

from backend.llm import LLM, LLMError, register_mock
from backend.state import ActivityBlock, ItineraryDay, ItineraryOutput, TravelState

#: A day of sightseeing before it stops being a holiday.
HOURS_PER_DAY = 6.5

#: Fallback when the traveller never said how long they have.
DEFAULT_DAYS = 3

ITINERARY_SYSTEM = """You build day-by-day travel schedules.

Return JSON:
  {"days": [{"day": 1, "theme": str,
             "blocks": [{"time": str, "title": str, "detail": str}]}]}

Rules:
- Produce exactly the number of days requested. No more, no fewer.
- Use ONLY the attractions listed in the prompt. You may add meals, rest and
  travel between them, but do not introduce sightseeing stops nobody researched.
- 3 to 5 blocks per day. A day of nothing but landmarks is exhausting.
- "time" is a clock time or a plain slot ("09:00", "Evening").
- Group stops that are near each other on the same day; do not bounce across a
  city and back.
- "theme" is a short phrase describing the day, e.g. "Old town and market food".
- "detail" is one practical sentence: what to book, when to arrive, what to skip."""

_SLOTS = ("09:00", "11:30", "13:00", "14:30", "17:00", "Evening")


@register_mock("itinerary")
def _mock_itinerary(context: dict[str, Any]) -> dict[str, Any]:
    """Greedy packer standing in for the model when no API key is set."""
    attractions = context.get("attractions") or []
    days_wanted = int(context.get("days") or DEFAULT_DAYS)

    # Fill each day up to the hour budget, then start the next one.
    buckets: list[list[dict[str, Any]]] = [[] for _ in range(days_wanted)]
    hours = [0.0] * days_wanted
    for item in attractions:
        target = min(range(days_wanted), key=lambda i: hours[i])
        if hours[target] >= HOURS_PER_DAY and any(b for b in buckets):
            continue
        buckets[target].append(item)
        hours[target] += float(item.get("est_hours") or 2.0)

    days = []
    for index, bucket in enumerate(buckets, start=1):
        categories = [b.get("category", "") for b in bucket]
        theme = max(set(categories), key=categories.count) if categories else "Free day"
        blocks = [
            {
                "time": _SLOTS[min(slot, len(_SLOTS) - 1)],
                "title": item.get("name", "Explore"),
                "detail": item.get("description", ""),
            }
            for slot, item in enumerate(bucket)
        ]
        if blocks:
            blocks.append(
                {"time": "13:00", "title": "Lunch nearby", "detail": "Keep it local and unhurried."}
            )
            blocks.sort(key=lambda b: _SLOTS.index(b["time"]) if b["time"] in _SLOTS else 2.5)
        days.append({"day": index, "theme": f"{theme.title()} focus", "blocks": blocks})

    return {"days": days}


def itinerary(state: TravelState, llm: LLM) -> dict[str, Any]:
    """Graph node: build the day-by-day schedule."""
    params = state["params"]
    assert params is not None

    research = state.get("research")
    attractions = research.attractions if research else []
    days_wanted = params.duration_days or DEFAULT_DAYS

    if not attractions:
        return {
            "itinerary": ItineraryOutput(days=[]),
            "errors": ["itinerary: no attractions in state; nothing to schedule"],
            "trace": ["itinerary"],
        }

    listing = "\n".join(
        f"- {a.name} ({a.category}, ~{a.est_hours:g}h): {a.description}" for a in attractions
    )

    try:
        raw = llm.json(
            task="itinerary",
            system=ITINERARY_SYSTEM,
            prompt=(
                f"Destination: {params.destination}\n"
                f"Days to fill: {days_wanted}\n"
                f"Travellers: {params.travelers}\n"
                f"Interests: {', '.join(params.preferences) or 'none stated'}\n\n"
                f"Attractions available:\n{listing}"
            ),
            context={
                "attractions": [a.model_dump() for a in attractions],
                "days": days_wanted,
            },
            max_tokens=4096,
        )
        errors: list[str] = []
    except LLMError as exc:
        return {
            "itinerary": ItineraryOutput(days=[]),
            "errors": [f"itinerary: {exc}"],
            "trace": ["itinerary"],
        }

    days: list[ItineraryDay] = []
    for entry in raw.get("days") or []:
        blocks = []
        for block in entry.get("blocks") or []:
            try:
                blocks.append(ActivityBlock(**block))
            except (TypeError, ValueError):
                continue
        try:
            days.append(
                ItineraryDay(day=int(entry.get("day", len(days) + 1)),
                             theme=str(entry.get("theme", "")),
                             blocks=blocks)
            )
        except (TypeError, ValueError):
            continue

    # The trip length is the one thing the traveller actually stated.
    if len(days) != days_wanted:
        errors.append(
            f"itinerary: asked for {days_wanted} days, model returned {len(days)}"
        )
        days = days[:days_wanted]

    return {"itinerary": ItineraryOutput(days=days), "errors": errors, "trace": ["itinerary"]}
