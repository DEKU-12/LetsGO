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
"""

from __future__ import annotations

import re
from typing import Any

from backend.agents.accommodation import accommodation
from backend.agents.aggregator import aggregator
from backend.agents.check import MAX_RETRIES, check_itinerary
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
- "instruction" restates the change as one direct instruction to the planner.
- "reply" is empty unless "agent" is {NONE}; then it is one friendly sentence
  saying why this cannot be done as an edit and what to do instead."""

_DAY = re.compile(r"\bday\s+(\d+)", re.IGNORECASE)
_DAYS = re.compile(r"\bdays\s+(\d+)\s+(?:and|&)\s+(\d+)", re.IGNORECASE)


@register_mock("edit_route")
def _mock_route(context: dict[str, Any]) -> dict[str, Any]:
    """Keyword router standing in for the model when no API key is set."""
    text = str(context.get("message", "")).lower()
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


def route_edit(state: TravelState, message: str, llm: LLM) -> dict[str, Any]:
    """Decide what an edit message is about. Always returns a usable decision."""
    params = state["params"]
    days_planned = len(state["itinerary"].days) if state.get("itinerary") else 0
    sections = [s for s in (ITINERARY, ACCOMMODATION, TRANSPORT) if state.get(s)]

    try:
        raw = llm.json(
            task="edit_route",
            system=EDIT_SYSTEM,
            prompt=(
                f"Trip: {params.destination}, {days_planned or 'no'} days planned\n"
                f"Sections in the plan: {', '.join(sections) or 'none'}\n\n"
                f"Change request: {message}"
            ),
            context={"message": message},
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


def edit_trip(state: TravelState, message: str, llm: LLM) -> tuple[TravelState | None, str]:
    """Apply one change to a finished plan.

    Returns ``(new_state, "")`` on success, or ``(None, reply)`` when the
    message is not an edit this plan can take — the reply says why.
    """
    state = TravelState(**state)  # never mutate the caller's copy
    state.update(trace=[], errors=[], itinerary_feedback=[], check_attempts=0, edit_request=None)

    decision = route_edit(state, message, llm)
    if decision["agent"] == NONE:
        return None, decision["reply"]

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

    apply_update(state, aggregator(state, llm))
    apply_update(state, validate_plan(state))
    return state, ""
