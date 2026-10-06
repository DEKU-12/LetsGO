

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


class Photo(BaseModel):
    """A freely licensed photo from Wikimedia Commons, with its credit.

    Commons licences (CC BY, CC BY-SA) require attribution, so the credit is
    part of the photo, not decoration.
    """

    url: str
    #: The Commons file page: author, licence, full size.
    page: str
    credit: str = ""
    #: Set when the photo shows something related rather than the place
    #: itself (a café linked to its famous pastry), so the plan says so.
    caption: str | None = None
    #: True when no photo of the place itself was found and this is a photo of
    #: the destination instead; the plan labels it as such.
    generic: bool = False


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
    #: From the same map entry that verified the place: its photo, and its
    #: opening hours in OpenStreetMap format ("Mo-Su 09:00-17:00").
    photo: Photo | None = None
    opening_hours: str | None = None


class WeatherOutlook(BaseModel):
    summary: str
    avg_high_c: float | None = None
    avg_low_c: float | None = None
    advice: str = ""


class ResearchOutput(BaseModel):
    destination: str
    attractions: list[Attraction] = Field(default_factory=list)
    weather: WeatherOutlook | None = None
    #: Outlook per city the research agent chose to check, for trips that
    #: cover several (Tokyo and Kyoto). `weather` is the first of these.
    city_weather: dict[str, WeatherOutlook] = Field(default_factory=dict)
    #: A photo of the destination itself, for the top of the plan.
    photo: Photo | None = None
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


class StayArea(BaseModel):
    """A neighbourhood to stay in. Advice about where, not a listing."""

    name: str
    why: str = ""
    #: Typical price level of places to stay there, from general knowledge.
    price_level: Budget | None = None
    #: Same meaning as Attraction.verified: confirmed on a map, not confirmed,
    #: or unchecked.
    verified: bool | None = None


class AccommodationOutput(BaseModel):
    areas: list[StayArea] = Field(default_factory=list)
    tips: list[str] = Field(default_factory=list)


class TransportTip(BaseModel):
    mode: str
    description: str


class TransportOutput(BaseModel):
    #: How travellers usually arrive: main airports, rail hubs.
    arrival: list[str] = Field(default_factory=list)
    local: list[TransportTip] = Field(default_factory=list)
    tips: list[str] = Field(default_factory=list)


class TravelState(TypedDict, total=False):
    """Shared state passed between every node in the graph."""

    # input
    request: str
    #: Lasting preferences the traveller asked us to remember (vegetarian,
    #: travels with kids). Every agent that plans something reads them.
    profile: list[str]

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

    # plan edits (see edit.py)
    #: A change the traveller asked for, read by the accommodation and
    #: transport agents when they are rerun on a finished plan.
    edit_request: str | None

    # aggregation
    final_plan: str | None

    # bookkeeping
    #: "<adapter>:<live|mock>" for every data source touched, so the finished
    #: plan can say which of its numbers are real.
    sources: Annotated[list[str], operator.add]
    trace: Annotated[list[str], operator.add]
    errors: Annotated[list[str], operator.add]
    meta: dict[str, Any]


def edit_note(state: TravelState) -> str:
    """The traveller's change request, when an agent is rerun to edit a plan."""
    request = state.get("edit_request")
    return f"\n\nThe traveller asked for this change: {request}" if request else ""


def profile_note(state: TravelState) -> str:
    """The traveller's saved preferences, as a line for an agent's prompt."""
    profile = state.get("profile") or []
    return f"\nAlways true for this traveller: {'; '.join(profile)}" if profile else ""


def new_state(request: str, profile: list[str] | None = None) -> TravelState:
    return TravelState(
        request=request,
        profile=list(profile or []),
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
        edit_request=None,
        final_plan=None,
        sources=[],
        trace=[],
        errors=[],
        meta={},
    )
