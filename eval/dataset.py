"""Labeled data the evaluation layer is scored against.

Two datasets live here:

* :data:`ROUTING_CASES` — requests annotated with the agents the supervisor
  *should* invoke. Hand-written, and deliberately not derived from what the
  system currently does, or the metric would only measure itself.
* Human plan grades — scores a person assigned to generated plans, stored on
  disk by ``python -m eval.grade``. These are what the LLM judge is validated
  against; without them the judge is unvalidated and the report says so.

Labelling rules used throughout:

* The aggregator is excluded — it always runs, so including it would inflate
  every score with a freebie.
* An agent is labelled only if the traveller asked for something that agent
  produces. "Where should I stay in Rome" needs accommodation, not transport.
* Itinerary implies research, because a schedule cannot be built without it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from backend.state import ACCOMMODATION, ITINERARY, RESEARCH, TRANSPORT

HUMAN_GRADES_PATH = Path(__file__).resolve().parent / "human_grades.json"


@dataclass(frozen=True)
class RoutingCase:
    """One request and the set of agents that ought to handle it."""

    id: str
    request: str
    expected: frozenset[str]
    why: str


def _case(id: str, request: str, expected: set[str], why: str) -> RoutingCase:
    return RoutingCase(id=id, request=request, expected=frozenset(expected), why=why)


ROUTING_CASES: tuple[RoutingCase, ...] = (
    # --- single agent -----------------------------------------------------
    _case("weather-rome", "What's the weather like in Rome in October?",
          {RESEARCH}, "A question about conditions. No trip is being planned."),
    _case("safety-marrakesh", "Is Marrakesh safe for a solo female traveller?",
          {RESEARCH}, "Practical destination information only."),
    _case("sights-lisbon", "What are the best things to see in Lisbon?",
          {RESEARCH}, "Attractions only; no schedule, stay or travel requested."),
    _case("visa-vietnam", "Do I need a visa for Vietnam and what's the best time to go?",
          {RESEARCH}, "Practical notes and seasonality — research alone."),
    _case("hotels-only-paris", "Find me a mid-range hotel in Paris near the Marais.",
          {ACCOMMODATION}, "Lodging only. No sightseeing or transport asked for."),
    _case("hostels-bangkok", "Cheap hostels in Bangkok for a backpacker?",
          {ACCOMMODATION}, "Lodging only."),
    _case("flights-tokyo", "What are my options for flying to Tokyo?",
          {TRANSPORT}, "Getting there only."),
    _case("getting-around-london", "How do I get around London without a car?",
          {TRANSPORT}, "Local transport only."),

    # --- two agents -------------------------------------------------------
    _case("itinerary-kyoto", "Plan me a 3-day sightseeing itinerary for Kyoto. "
                             "I've already booked my hotel and flights.",
          {RESEARCH, ITINERARY},
          "A schedule is wanted; stay and travel are explicitly handled."),
    _case("weekend-barcelona", "I have a free weekend in Barcelona, what should I do each day?",
          {RESEARCH, ITINERARY}, "Day-by-day plan; nothing about beds or flights."),
    _case("stay-and-see-athens", "Where should I stay in Athens and what's worth visiting?",
          {RESEARCH, ACCOMMODATION}, "Two explicit asks, no schedule requested."),
    _case("flights-and-hotel-reykjavik",
          "I need flights and a hotel for four nights in Reykjavik.",
          {ACCOMMODATION, TRANSPORT}, "Logistics only; no sightseeing asked for."),

    # --- three agents -----------------------------------------------------
    _case("trip-no-flights-porto",
          "Plan 4 days in Porto with somewhere to stay. I'm driving there myself.",
          {RESEARCH, ITINERARY, ACCOMMODATION},
          "Full plan minus transport, which the traveller has ruled out."),
    _case("trip-no-hotel-oslo",
          "Plan a 5-day trip to Oslo and how to get there. I'm staying with family.",
          {RESEARCH, ITINERARY, TRANSPORT},
          "Full plan minus accommodation, which is already arranged."),

    # --- everything -------------------------------------------------------
    _case("full-japan", "5 days in Japan, mid-range budget, love food and history. "
                        "I need hotels and flights too.",
          {RESEARCH, ITINERARY, ACCOMMODATION, TRANSPORT},
          "Explicitly asks for all four."),
    _case("full-peru", "Plan me a 10-day trip to Peru on a budget — everything, "
                       "start to finish.",
          {RESEARCH, ITINERARY, ACCOMMODATION, TRANSPORT},
          "'Everything, start to finish' is the whole team."),
    _case("honeymoon-greece",
          "Two of us, 7 nights in Greece for our honeymoon, luxury budget. "
          "Sort the whole thing out.",
          {RESEARCH, ITINERARY, ACCOMMODATION, TRANSPORT},
          "Complete trip planning."),

    # --- should not plan at all -------------------------------------------
    _case("nonsense-maths", "what is 2 + 2",
          set(), "Not a travel request. The guard should ask for clarification."),
    _case("no-destination", "I want to go somewhere warm next month, any ideas?",
          set(), "No destination to research. Should ask which place, not guess one."),
)


@dataclass
class HumanGrade:
    """One plan, scored by a person, for validating the LLM judge."""

    case_id: str
    request: str
    plan: str
    scores: dict[str, int] = field(default_factory=dict)
    note: str = ""

    @property
    def overall(self) -> float:
        return sum(self.scores.values()) / len(self.scores) if self.scores else 0.0


def load_human_grades() -> list[HumanGrade]:
    """Read the grades a person recorded with `python -m eval.grade`."""
    if not HUMAN_GRADES_PATH.exists():
        return []
    raw = json.loads(HUMAN_GRADES_PATH.read_text())
    return [HumanGrade(**entry) for entry in raw]


def save_human_grades(grades: list[HumanGrade]) -> None:
    HUMAN_GRADES_PATH.write_text(
        json.dumps([g.__dict__ for g in grades], indent=2, ensure_ascii=False)
    )
