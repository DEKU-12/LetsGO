"""Input and output guards.

Input: turn free text into a validated :class:`TripParams` before any agent
runs, so a nonsensical request is answered with a clarifying question instead
of being passed downstream as garbage.

Output: check the aggregator produced something well formed before it is
returned to the user.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import ValidationError

from backend.llm import LLM, LLMError, register_mock
from backend.state import TravelState, TripParams

PARSE_SYSTEM = """You extract structured trip parameters from a traveller's request.

Return a JSON object with exactly these keys:
  destination   string  - city or region; "" if the request names no place
  duration_days integer or null
  start_date    string (YYYY-MM-DD) or null
  budget        one of "budget", "mid-range", "luxury", or null
  travelers     integer, default 1
  preferences   array of short lowercase strings (e.g. ["food", "history"])
  notes         string or null - anything else that constrains the trip
  usable        boolean - false if this is not a travel request at all
  clarification string or null - if usable is false, one short question to ask

Infer nothing that is not implied. Do not invent a destination."""

_DURATION = re.compile(r"(\d+)\s*(?:day|night)s?", re.I)
_TRAVELERS = re.compile(r"(\d+)\s*(?:people|persons?|travell?ers?|adults?)", re.I)
# Checked in order — the more specific phrasings first, so "mid-range budget"
# is not read as a shoestring trip.
_BUDGET_WORDS = {
    "luxury": ("luxury", "luxurious", "five star", "5 star", "high end", "splurge"),
    "mid-range": ("mid-range", "midrange", "mid range", "moderate", "comfortable"),
    "budget": ("budget", "cheap", "backpack", "hostel", "shoestring", "affordable"),
}
_PREF_WORDS = (
    "food", "history", "art", "nature", "hiking", "museums", "nightlife",
    "shopping", "beaches", "architecture", "photography", "family", "temples",
    "wine", "coffee", "music", "sports", "relaxing",
)
_KNOWN_PLACES = (
    "tokyo", "kyoto", "osaka", "japan", "paris", "france", "rome", "italy",
    "barcelona", "spain", "london", "reykjavik", "iceland", "bangkok",
    "thailand", "lisbon", "portugal", "new york", "marrakesh", "morocco",
    "peru", "vietnam", "greece", "norway",
)


@register_mock("parse_request")
def _mock_parse(context: dict[str, Any]) -> dict[str, Any]:
    """Keyword parser standing in for the model when no API key is set."""
    text = str(context.get("request", ""))
    low = text.lower()

    destination = next((p for p in _KNOWN_PLACES if p in low), "")
    duration = _DURATION.search(low)
    travelers = _TRAVELERS.search(low)

    budget = None
    for level, words in _BUDGET_WORDS.items():
        if any(w in low for w in words):
            budget = level
            break

    usable = bool(destination) or "trip" in low or "travel" in low
    return {
        "destination": destination.title(),
        "duration_days": int(duration.group(1)) if duration else None,
        "start_date": None,
        "budget": budget,
        "travelers": int(travelers.group(1)) if travelers else 1,
        "preferences": [w for w in _PREF_WORDS if w in low],
        "notes": None,
        "usable": usable,
        "clarification": None
        if usable
        else "Which destination would you like to plan a trip to?",
    }


def parse_request(state: TravelState, llm: LLM) -> dict[str, Any]:
    """Graph node: free text in, validated TripParams (or a clarification) out."""
    request = state["request"]

    try:
        raw = llm.json(
            task="parse_request",
            system=PARSE_SYSTEM,
            prompt=f"Traveller request:\n{request}",
            context={"request": request},
            max_tokens=1024,
        )
    except LLMError as exc:
        return {
            "clarification": "Sorry — I couldn't read that request. Could you rephrase it?",
            "errors": [f"parse_request: {exc}"],
            "trace": ["guard:parse_request"],
        }

    if not raw.get("usable", True) or not str(raw.get("destination", "")).strip():
        return {
            "clarification": raw.get("clarification")
            or "Which destination would you like to plan a trip to?",
            "trace": ["guard:parse_request"],
        }

    try:
        params = TripParams(
            destination=str(raw["destination"]).strip(),
            duration_days=raw.get("duration_days"),
            start_date=raw.get("start_date"),
            budget=raw.get("budget") if raw.get("budget") in {"budget", "mid-range", "luxury"} else None,
            travelers=int(raw.get("travelers") or 1),
            preferences=[str(p).lower() for p in raw.get("preferences") or []],
            notes=raw.get("notes"),
        )
    except (ValidationError, TypeError, ValueError) as exc:
        return {
            "clarification": "I couldn't pin down the trip details. Could you restate the destination and length?",
            "errors": [f"parse_request: {exc}"],
            "trace": ["guard:parse_request"],
        }

    return {"params": params, "trace": ["guard:parse_request"]}


def validate_plan(state: TravelState) -> dict[str, Any]:
    """Graph node: last check that we are handing back something usable."""
    plan = state.get("final_plan")
    problems: list[str] = []

    if not plan or len(plan.strip()) < 80:
        problems.append("final plan is missing or too short to be useful")
    elif "#" not in plan:
        problems.append("final plan has no headings — expected structured markdown")

    if problems:
        return {
            "errors": [f"validate_plan: {p}" for p in problems],
            "trace": ["guard:validate_plan"],
        }
    return {"trace": ["guard:validate_plan"]}
