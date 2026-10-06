"""Change a finished plan by chatting — "less walking on day 3".

An edit reruns only the part of the plan it is about, not the whole graph:

1. A small model call routes the message: which section (itinerary,
   accommodation or transport), which days, and the change as an instruction.
   Requests that would change the whole trip (a new destination, a different
   length) are declined with a reply rather than half-applied.
2. That one agent reruns with the previous version and the instruction.
3. For the itinerary, days the traveller did not mention are put back exactly
   as they were. This is enforced in code, not asked of the model: "change
   nothing else" in a prompt is a request, and a plan that silently rewrites
   day 1 while fixing day 3 is the failure people notice.
4. The schedule check runs as it does for a new plan, then the aggregator
   writes the plan again.

The caller saves the result as a new version, so an edit can always be undone.

**Trip-day help** is the same machinery with two facts added: which day of the
trip it is, so "rearrange today" reaches the right day, and the weather right
now. ``check_today`` looks at the live weather and, if it is bad for being
outside, rearranges today without being asked; if it is fine, it says so and
changes nothing.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Any

from backend.agents.accommodation import accommodation
from backend.agents.aggregator import aggregator
from backend.agents.check import MAX_RETRIES, match_attraction, check_itinerary
from backend.agents.itinerary import itinerary
from backend.agents.transport import transport
from backend.graph import apply_update
from backend.guards import validate_plan
from backend.llm import LLM, LLMError, register_mock
from backend.state import ACCOMMODATION, ITINERARY, TRANSPORT, ItineraryOutput, TravelState

NONE = "none"

EDIT_SYSTEM = f"""You route a traveller's change request on a travel plan they already have.

Return JSON:
  {{"agent": str, "days": [int], "instruction": str, "reply": str}}

"agent" is one of:
  {ITINERARY}     - what happens on which day: stops, pace, timing, meals, order
  {ACCOMMODATION} - where they stay
  {TRANSPORT}     - getting there and getting around
  {NONE}          - anything else: a different destination or trip length (that is a
                 new plan, not an edit), a question rather than a change, or nonsense

Rules:
- "days" lists the day numbers the change is about. Use [] when it applies to the
  whole trip, and always [] unless "agent" is {ITINERARY}. "the last day" is the
  final day number you are given.
- If you are told which day of the trip it is today, "today", "this morning" and
  "tonight" mean that day and "tomorrow" means the day after.
- "instruction" restates the change as one direct instruction to the planner.
- "reply" is empty unless "agent" is {NONE}; then it is one friendly sentence
  saying why this cannot be done as an edit and what to do instead."""

_DAY = re.compile(r"\bday\s+(\d+)", re.IGNORECASE)
_DAYS = re.compile(r"\bdays\s+(\d+)\s+(?:and|&)\s+(\d+)", re.IGNORECASE)


@register_mock("edit_route")
def _mock_route(context: dict[str, Any]) -> dict[str, Any]:
    """Keyword router standing in for the model when no API key is set."""
    text = str(context.get("message", "")).lower()
    if context.get("today"):
        today = int(context["today"])
        text = text.replace("tomorrow", f"day {today + 1}").replace("today", f"day {today}")
    if any(w in text for w in ("hotel", "stay", "hostel", "room")):
        return {"agent": ACCOMMODATION, "days": [], "instruction": text, "reply": ""}
    if any(w in text for w in ("taxi", "train", "flight", "bus", "transport", "metro")):
        return {"agent": TRANSPORT, "days": [], "instruction": text, "reply": ""}
    if any(w in text for w in ("instead", "different country", "?")):
        return {"agent": NONE, "days": [], "instruction": "",
                "reply": "That changes the whole trip, so start a new plan for it."}
    pair = _DAYS.search(text)
    days = [int(d) for d in pair.groups()] if pair else [int(d) for d in _DAY.findall(text)]
    return {"agent": ITINERARY, "days": days, "instruction": text, "reply": ""}


def route_edit(
    state: TravelState, message: str, llm: LLM, today: int | None = None
) -> dict[str, Any]:
    """Decide what an edit message is about. Always returns a usable decision.

    `today` is the day of the trip the traveller is on, if known, so "today"
    and "tomorrow" in the message can be resolved to day numbers.
    """
    params = state["params"]
    days_planned = len(state["itinerary"].days) if state.get("itinerary") else 0
    sections = [s for s in (ITINERARY, ACCOMMODATION, TRANSPORT) if state.get(s)]

    try:
        raw = llm.json(
            task="edit_route",
            system=EDIT_SYSTEM,
            prompt=(
                f"Trip: {params.destination}, {days_planned or 'no'} days planned\n"
                f"Sections in the plan: {', '.join(sections) or 'none'}\n"
                + (f"Today is day {today} of the trip.\n" if today else "")
                + f"\nChange request: {message}"
            ),
            context={"message": message, "today": today},
            max_tokens=512,
        )
    except LLMError as exc:
        return {"agent": NONE, "days": [], "instruction": "",
                "reply": "Sorry, I couldn't process that change just now. Try again?",
                "error": str(exc)}

    agent = raw.get("agent") if raw.get("agent") in (ITINERARY, ACCOMMODATION, TRANSPORT) else NONE
    days = sorted({int(d) for d in raw.get("days") or [] if str(d).isdigit()})
    days = [d for d in days if 1 <= d <= days_planned] if agent == ITINERARY else []
    if len(days) == days_planned:
        days = []  # every day named is the same as the whole trip
    reply = str(raw.get("reply") or "")

    # An itinerary needs researched places to schedule; a plan that never had
    # any (a weather question) cannot grow one through an edit.
    if agent == ITINERARY and not (state.get("research") and state["research"].attractions):
        agent, reply = NONE, "This plan has no places to schedule yet. Ask for a new trip plan instead."
    if agent == NONE and not reply:
        reply = "That isn't something I can change in this plan. Try starting a new plan."

    return {
        "agent": agent,
        "days": days,
        "instruction": str(raw.get("instruction") or message),
        "reply": reply,
    }


def keep_untouched_days(
    old: ItineraryOutput | None, new: ItineraryOutput, days: list[int]
) -> ItineraryOutput:
    """Take the edited days from `new`, and every other day from `old`."""
    if not days or old is None:
        return new
    edited = {d.day: d for d in new.days}
    return ItineraryOutput(
        days=[edited.get(d.day, d) if d.day in days else d for d in old.days]
    )


FAILED_REPLY = "Sorry, I couldn't make that change just now. Nothing was changed; try again in a minute."


def _model_failed(update: dict[str, Any], agent: str, llm: LLM) -> bool:
    """Did this agent's model call fail, leaving it to fall back?

    Agents catch LLMError and record it as ``"<agent>: <error>"``. A provider
    failure reads ``"<agent>: <provider>: ..."`` and a JSON failure
    ``"<agent>: <task>: model did not return valid JSON"`` (see llm.py). On a
    new plan the fallback is fine; on an edit it would mean saving the old
    section and calling it changed.
    """
    return any(
        e.startswith((f"{agent}: {llm.provider}:", f"{agent}: {agent}:"))
        for e in update.get("errors") or []
    )


def _rerun_itinerary(
    state: TravelState, llm: LLM, days: list[int], feedback: list[str]
) -> bool:
    """Redo the schedule. Returns False, changing nothing, if the model failed."""
    previous = state.get("itinerary")
    state["itinerary_feedback"] = feedback
    update = itinerary(state, llm)
    if _model_failed(update, ITINERARY, llm):
        return False
    update["itinerary"] = keep_untouched_days(previous, update["itinerary"], days)
    apply_update(state, update)
    return True


#: Categories (from the research agent) that are mostly outdoors.
OUTDOOR = {"nature", "neighbourhood"}


def trip_day(state: TravelState, on: date) -> int | None:
    """Which day of the trip `on` is, if the plan has a start date."""
    params = state.get("params")
    itinerary_ = state.get("itinerary")
    if not params or not params.start_date or not itinerary_:
        return None
    try:
        start = date.fromisoformat(params.start_date)
    except ValueError:
        return None
    day = (on - start).days + 1
    return day if 1 <= day <= len(itinerary_.days) else None


def weather_brief(state: TravelState, day: int, conditions: dict[str, Any]) -> str:
    """Today's weather, plus the facts the planner needs to work around it.

    Which of today's stops are outdoors, and which indoor places the research
    found that are not scheduled anywhere yet — so a rainy-day swap can use
    real alternatives instead of inventing them.
    """
    research = state.get("research")
    itinerary_ = state.get("itinerary")
    if not research or not itinerary_:
        return ""
    attractions = research.attractions

    today = next((d for d in itinerary_.days if d.day == day), None)
    today_stops = [a for a in (match_attraction(b.title, attractions) for b in today.blocks) if a] if today else []
    scheduled = {
        a.name
        for d in itinerary_.days
        for a in (match_attraction(b.title, attractions) for b in d.blocks)
        if a
    }
    outdoor = [a.name for a in today_stops if a.category in OUTDOOR]
    spare = [f"{a.name} ({a.category})" for a in attractions
             if a.name not in scheduled and a.category not in OUTDOOR]

    return (
        f" Weather right now: {conditions['description']}, {conditions['temp_c']:g}°C."
        + (f" Outdoor stops on day {day}: {', '.join(outdoor)}." if outdoor else "")
        + (f" Researched places not scheduled on any day: {', '.join(spare)}." if spare else "")
    )


def edit_trip(
    state: TravelState,
    message: str,
    llm: LLM,
    today: int | None = None,
    conditions: dict[str, Any] | None = None,
) -> tuple[TravelState | None, str]:
    """Apply one change to a finished plan.

    `today` (the day of the trip the traveller is on) and `conditions` (live
    weather, from the weather adapter with ``when="now"``) are optional; with
    them, "it's raining, rearrange today" reaches the right day and the planner
    knows the actual weather.

    Returns ``(new_state, "")`` on success, or ``(None, reply)`` when the
    message is not an edit this plan can take — the reply says why.
    """
    decision = route_edit(state, message, llm, today)
    if decision["agent"] == NONE:
        return None, decision["reply"]
    if (
        decision["agent"] == ITINERARY
        and today in (decision["days"] or [today])
        and conditions
        and conditions.get("description")
    ):
        decision["instruction"] += weather_brief(state, today, conditions)
        decision["weather"] = conditions
    return _apply(state, decision, message, llm)


def check_today(
    state: TravelState, today: int, conditions: dict[str, Any], llm: LLM
) -> tuple[TravelState | None, str]:
    """Look at today's weather and rearrange today only if it calls for it.

    No model call decides whether to act: "is it raining" is a fact from the
    weather service, not a judgement. Returns ``(None, reply)`` when nothing
    needs changing or the weather is unknown.
    """
    if not conditions.get("description"):
        return None, "Live weather isn't available (no OpenWeather key is set), so I can't check today."
    summary = f"{conditions['description']}, {conditions['temp_c']:g}°C"
    if not conditions.get("bad"):
        return None, f"It's {summary} — day {today} works as planned."
    if not (state.get("research") and state["research"].attractions):
        return None, f"It's {summary}, but this plan has no schedule to rearrange."

    decision = {
        "agent": ITINERARY,
        "days": [today],
        "instruction": (
            f"Rearrange day {today} for the weather: replace outdoor stops with indoor "
            "ones, keeping the day's pace and meals." + weather_brief(state, today, conditions)
        ),
        "reply": "",
        "weather": conditions,
    }
    return _apply(state, decision, f"Check today (day {today})", llm)


def _apply(
    state: TravelState, decision: dict[str, Any], message: str, llm: LLM
) -> tuple[TravelState | None, str]:
    """Rerun the one agent a routed edit is about, then rewrite the plan."""
    before = state.get(decision["agent"])
    state = TravelState(**state)  # never mutate the caller's copy
    state.update(trace=[], errors=[], itinerary_feedback=[], check_attempts=0, edit_request=None)

    meta = dict(state.get("meta") or {})
    meta["edit"] = {"message": message, **decision}
    meta.pop("itinerary_checks", None)
    apply_update(state, {"meta": meta, "trace": ["edit_router"]})

    agent, days = decision["agent"], decision["days"]
    if agent == ITINERARY:
        scope = (
            f" Only change day{'s' if len(days) > 1 else ''} "
            f"{', '.join(map(str, days))}; keep every other day exactly as it is."
            if days else ""
        )
        if not _rerun_itinerary(state, llm, days, [decision["instruction"] + scope]):
            return None, FAILED_REPLY
        # The same check a new plan gets, with the same retry budget. A failed
        # retry keeps the edited version and records what is still wrong.
        while True:
            apply_update(state, check_itinerary(state))
            problems = state.get("itinerary_feedback") or []
            if not problems or state["check_attempts"] > MAX_RETRIES:
                break
            if not _rerun_itinerary(state, llm, days, problems):
                break
    else:
        state["edit_request"] = decision["instruction"]
        node = accommodation if agent == ACCOMMODATION else transport
        update = node(state, llm)
        if _model_failed(update, agent, llm):
            return None, FAILED_REPLY
        apply_update(state, update)

    # Did anything actually change? A model can reasonably decide a day is
    # already fine for the weather; the reply should say so, not "updated".
    state["meta"]["edit"]["changed"] = state.get(agent) != before

    apply_update(state, aggregator(state, llm))
    apply_update(state, validate_plan(state))
    return state, ""
