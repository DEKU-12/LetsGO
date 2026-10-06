

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

from pydantic import BaseModel, Field

# Worker agent names. These strings are the vocabulary the supervisor routes
# with and the labels the eval dataset is annotated against — keep them stable.
RESEARCH = "destination_research"
ITINERARY = "itinerary"
ACCOMMODATION = "accommodation"
TRANSPORT = "transport"
AGGREGATOR = "aggregator"

WORKERS: tuple[str, ...] = (RESEARCH, ITINERARY, ACCOMMODATION, TRANSPORT, AGGREGATOR)

Budget = Literal["budget", "mid-range", "luxury"]


class TripParams(BaseModel):
    """The user's free-text request, parsed into something agents can use."""

    destination: str
    duration_days: int | None = None
    start_date: str | None = None
    budget: Budget | None = None
    travelers: int = 1
    preferences: list[str] = Field(default_factory=list)
    notes: str | None = None


class Attraction(BaseModel):
    name: str
    category: str = "attraction"
    description: str = ""
    est_hours: float = 2.0
    #: True if an independent geographic dataset confirmed this place exists at
    #: the destination, False if it could not be confirmed, None if unchecked.
    #: False means "not confirmed", not "fake" — see adapters/places.py.
    verified: bool | None = None
    #: Map position, filled in when the place verifier confirmed it. Lets the
    #: schedule check measure how far apart one day's stops are.
    lat: float | None = None
    lon: float | None = None


class WeatherOutlook(BaseModel):
    summary: str
    avg_high_c: float | None = None
    avg_low_c: float | None = None
    advice: str = ""


class ResearchOutput(BaseModel):
    destination: str
    attractions: list[Attraction] = Field(default_factory=list)
    weather: WeatherOutlook | None = None
    practical_notes: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)


class ActivityBlock(BaseModel):
    time: str
    title: str
    detail: str = ""


class ItineraryDay(BaseModel):
    day: int
    theme: str = ""
    blocks: list[ActivityBlock] = Field(default_factory=list)


class ItineraryOutput(BaseModel):
    days: list[ItineraryDay] = Field(default_factory=list)


class LodgingOption(BaseModel):
    name: str
    area: str = ""
    price_per_night_usd: float = 0.0
    rating: float | None = None
    why: str = ""


class AccommodationOutput(BaseModel):
    options: list[LodgingOption] = Field(default_factory=list)
    nightly_budget_usd: float | None = None


class TransportLeg(BaseModel):
    mode: str
    description: str
    est_cost_usd: float | None = None


class TransportOutput(BaseModel):
    inbound: list[TransportLeg] = Field(default_factory=list)
    local: list[TransportLeg] = Field(default_factory=list)


class TravelState(TypedDict, total=False):
    """Shared state passed between every node in the graph."""

    # input
    request: str

    # guards + supervisor
    params: TripParams | None
    route_plan: list[str]
    cursor: int
    clarification: str | None

    # worker outputs
    research: ResearchOutput | None
    itinerary: ItineraryOutput | None
    accommodation: AccommodationOutput | None
    transport: TransportOutput | None

    # schedule check (see agents/check.py)
    #: Problems the last check found; the itinerary agent reads these on a redo.
    itinerary_feedback: list[str]
    check_attempts: int

    # aggregation
    final_plan: str | None

    # bookkeeping
    #: "<adapter>:<live|mock>" for every data source touched, so the finished
    #: plan can say which of its numbers are real.
    sources: Annotated[list[str], operator.add]
    trace: Annotated[list[str], operator.add]
    errors: Annotated[list[str], operator.add]
    meta: dict[str, Any]


def new_state(request: str) -> TravelState:
    return TravelState(
        request=request,
        params=None,
        route_plan=[],
        cursor=0,
        clarification=None,
        research=None,
        itinerary=None,
        accommodation=None,
        transport=None,
        itinerary_feedback=[],
        check_attempts=0,
        final_plan=None,
        sources=[],
        trace=[],
        errors=[],
        meta={},
    )
