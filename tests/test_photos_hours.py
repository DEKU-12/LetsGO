"""Photos of places, and opening hours: reading, checking, showing."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from backend.adapters.wikimedia import WikimediaAdapter
from backend.agents.aggregator import aggregator
from backend.agents.check import find_problems
from backend.agents.destination_research import _photo_for
from backend.hours import open_at, parse
from backend.llm import LLM
from backend.state import (
    ActivityBlock,
    Attraction,
    ItineraryDay,
    ItineraryOutput,
    Photo,
    ResearchOutput,
    TripParams,
)

# -- reading hours ------------------------------------------------------------


@pytest.mark.parametrize(("value", "day", "minute", "expected"), [
    ("Mo-Su 09:00-17:00", None, 10 * 60, True),
    ("Mo-Su 09:00-17:00", None, 18 * 60, False),
    ("Tu-Su 10:00-18:00; Mo off", 0, 11 * 60, False),
    ("Tu-Su 10:00-18:00; Mo off", 1, 11 * 60, True),
    ("Mo-Fr 09:00-12:00,13:00-17:00", 2, 12 * 60 + 30, False),
    ("24/7", 3, 3 * 60, True),
    ("Fr-Sa 18:00-02:00", 4, 23 * 60, True),
])
def test_common_hours_are_read(value, day, minute, expected) -> None:
    assert open_at(parse(value), day, minute) is expected


@pytest.mark.parametrize("value", [
    "Mar 30-Sep 30: 08:30-19:15; Oct 01-Feb 28: 08:30-16:30",
    "sunrise-sunset",
    'Mo-Fr 09:00-17:00 "by appointment"',
    "",
    None,
])
def test_unusual_hours_are_left_unread_not_guessed(value) -> None:
    assert parse(value) is None


# -- checking a schedule against them -----------------------------------------


def _state(blocks_per_day: list[list[tuple[str, str]]], attractions, start_date=None):
    return {
        "params": TripParams(destination="Kyoto", duration_days=len(blocks_per_day),
                             start_date=start_date),
        "research": ResearchOutput(destination="Kyoto", attractions=attractions),
        "itinerary": ItineraryOutput(days=[
            ItineraryDay(day=i, blocks=[ActivityBlock(time=t, title=n) for t, n in blocks])
            for i, blocks in enumerate(blocks_per_day, start=1)
        ]),
    }


TEMPLE = Attraction(name="Kinkaku-ji", opening_hours="Mo-Su 09:00-17:00")
MUSEUM = Attraction(name="National Museum", opening_hours="Tu-Su 09:30-17:00; Mo off")
ODD = Attraction(name="Colosseum", opening_hours="Mar 30-Sep 30: 08:30-19:15")


def test_a_visit_after_closing_is_flagged() -> None:
    problems = find_problems(_state([[("18:00", "Kinkaku-ji")]], [TEMPLE]))
    assert any("outside its opening hours" in p for p in problems)


def test_a_closed_weekday_is_flagged_only_on_dated_trips() -> None:
    plan = [[("10:00", "National Museum")]]
    # 2026-10-05 is a Monday.
    dated = find_problems(_state(plan, [MUSEUM], start_date="2026-10-05"))
    assert any("closed on Mondays" in p for p in dated)
    assert find_problems(_state(plan, [MUSEUM])) == []  # weekday unknown: not judged


def test_unreadable_hours_and_vague_slots_are_not_judged() -> None:
    plan = [[("23:00", "Colosseum"), ("Evening", "Kinkaku-ji")]]
    assert find_problems(_state(plan, [ODD, TEMPLE])) == []


def test_a_visit_within_hours_passes() -> None:
    assert find_problems(_state([[("10:00", "Kinkaku-ji")]], [TEMPLE])) == []


# -- photos -------------------------------------------------------------------


def test_a_photo_of_something_related_gets_a_caption() -> None:
    details = {"Pastéis de Belém": {"wikidata": "Q1"}, "Colosseum": {"wikidata": "Q2"}}
    photos = {
        "Q1": {"url": "https://thumb.wikimedia.org/a.jpg", "page": "p", "credit": "c",
               "label": "pastel de nata"},
        "Q2": {"url": "https://thumb.wikimedia.org/b.jpg", "page": "p", "credit": "c",
               "label": "Colosseum"},
    }
    assert _photo_for("Pastéis de Belém", details, photos).caption == "pastel de nata"
    assert _photo_for("Colosseum", details, photos).caption is None
    assert _photo_for("Unknown place", details, photos) is None


def test_the_plan_shows_photos_hours_and_credits() -> None:
    photo = Photo(url="https://thumb.wikimedia.org/x_(1).jpg",
                  page="https://commons.wikimedia.org/wiki/File:x_(1).jpg",
                  credit="Pitan, CC BY-SA 3.0", caption="pastel de nata")
    state = {
        "params": TripParams(destination="Lisbon", duration_days=2),
        "research": ResearchOutput(destination="Lisbon", photo=photo, attractions=[
            Attraction(name="Pastéis de Belém", opening_hours="Mo-Su 08:00-21:00", photo=photo),
        ]),
    }
    plan = aggregator(state, LLM(provider="mock"))["final_plan"]

    assert "![Lisbon](https://thumb.wikimedia.org/x_%281%29.jpg)" in plan
    assert "open Mo-Su 08:00-21:00" in plan
    assert "Photo: Pitan, CC BY-SA 3.0" in plan
    assert "shows pastel de nata" in plan


def _fake_get(responses: dict[str, dict[str, Any]]):
    def get(self, url: str, params=None, **_: Any) -> Any:
        body = responses["wikidata" if "wikidata" in url else "commons"]
        return SimpleNamespace(json=lambda: body, raise_for_status=lambda: None)
    return get


def test_wikimedia_turns_ids_into_credited_photos(monkeypatch) -> None:
    monkeypatch.setattr(httpx.Client, "get", _fake_get({
        "wikidata": {"entities": {"Q10285": {
            "labels": {"en": {"value": "Colosseum"}},
            "claims": {"P18": [{"mainsnak": {"datavalue": {"value": "Colosseo 2020.jpg"}}}]},
        }}},
        "commons": {"query": {
            "normalized": [{"from": "File:Colosseo 2020.jpg", "to": "File:Colosseo 2020.jpg"}],
            "pages": {"1": {"title": "File:Colosseo 2020.jpg", "imageinfo": [{
                "thumburl": "https://thumb.wikimedia.org/c.jpg?utm_source=x",
                "descriptionurl": "https://commons.wikimedia.org/wiki/File:Colosseo_2020.jpg",
                "extmetadata": {"Artist": {"value": "<a href='u'>FeaturedPics</a>"},
                                "LicenseShortName": {"value": "CC BY-SA 4.0"}},
            }]}},
        }},
    }))

    photo = WikimediaAdapter().fetch(qids=["Q10285", "not-an-id"]).data["Q10285"]

    assert photo["url"] == "https://thumb.wikimedia.org/c.jpg"  # tracking query dropped
    assert photo["credit"] == "FeaturedPics, CC BY-SA 4.0"  # HTML stripped
    assert photo["label"] == "Colosseum"


def test_images_from_anywhere_but_wikimedia_are_refused(monkeypatch) -> None:
    monkeypatch.setattr(httpx.Client, "get", _fake_get({
        "wikidata": {"entities": {"Q1": {
            "labels": {}, "claims": {"P18": [{"mainsnak": {"datavalue": {"value": "a.jpg"}}}]},
        }}},
        "commons": {"query": {"pages": {"1": {"title": "File:a.jpg", "imageinfo": [{
            "thumburl": "https://tracker.example.com/a.jpg"}]}}}},
    }))
    assert WikimediaAdapter().fetch(qids=["Q1"]).data == {}


# -- photos for every place ---------------------------------------------------

from backend.adapters.places import name_variants  # noqa: E402
from backend.agents.destination_research import attach_photos  # noqa: E402

MIAMI = (25.78, -80.13)


def test_combined_names_are_split_but_generic_halves_are_not() -> None:
    assert "South Beach" in name_variants("South Beach and Ocean Drive")
    assert "Ocean Drive" in name_variants("South Beach and Ocean Drive")
    assert "Shark Valley" in name_variants("Everglades National Park – Shark Valley")
    # "Gardens" alone would match any garden in the city.
    assert "Gardens" not in name_variants("Vizcaya Museum and Gardens")


def _search_results(pages: list[dict[str, Any]]):
    responses = {
        "wikipedia": {"query": {"pages": {str(i): p for i, p in enumerate(pages)}}},
        "commons": {"query": {"pages": {
            str(i): {"title": f"File:{p['pageprops']['page_image_free']}",
                     "imageinfo": [{"thumburl": f"https://upload.wikimedia.org/{i}.jpg"}]}
            for i, p in enumerate(pages)
        }}},
    }

    def get(self, url: str, params=None, **_: Any) -> Any:
        body = responses["wikipedia" if "wikipedia" in url else "commons"]
        return SimpleNamespace(json=lambda: body, raise_for_status=lambda: None)
    return get


def _page(title: str, lat: float, lon: float, index: int) -> dict[str, Any]:
    return {"title": title, "index": index, "pageprops": {"page_image_free": f"{index}.jpg"},
            "coordinates": [{"lat": lat, "lon": lon}]}


def test_wikipedia_fallback_needs_the_right_name_in_the_right_place(monkeypatch) -> None:
    monkeypatch.setattr(httpx.Client, "get", _search_results([
        _page("Little Havana, Calgary", 51.0, -114.0, 1),   # same name, wrong continent
        _page("Miami Beach", 25.79, -80.13, 2),             # right place, wrong name
        _page("Little Havana", 25.77, -80.22, 3),           # both right
    ]))
    found = WikimediaAdapter().fetch(searches=[{
        "key": "Little Havana", "query": "Little Havana Miami",
        "names": ["Little Havana"], "lat": MIAMI[0], "lon": MIAMI[1],
    }]).data
    assert found["Little Havana"]["label"] == "Little Havana"
    assert found["Little Havana"]["url"] == "https://upload.wikimedia.org/2.jpg"


def test_a_name_match_far_away_is_rejected(monkeypatch) -> None:
    monkeypatch.setattr(httpx.Client, "get", _search_results([
        _page("Little Havana", 51.0, -114.0, 1),
    ]))
    found = WikimediaAdapter().fetch(searches=[{
        "key": "Little Havana", "query": "Little Havana Miami",
        "names": ["Little Havana"], "lat": MIAMI[0], "lon": MIAMI[1],
    }]).data
    assert found == {}


def test_places_without_a_photo_get_a_labelled_city_photo(monkeypatch) -> None:
    city = {"url": "https://upload.wikimedia.org/miami.jpg", "page": "p", "credit": "c",
            "label": "Miami"}
    monkeypatch.setattr(
        WikimediaAdapter, "fetch",
        lambda self, qids=(), searches=(): SimpleNamespace(
            data={"Q1": city} if qids else {}, source="live", provider="wikimedia"),
    )
    places = [Attraction(name="Lunch on Washington Avenue", category="food")]

    destination = attach_photos(places, "Miami", {"Miami": {"wikidata": "Q1"}}, None, set())

    assert destination is not None and not destination.generic
    assert places[0].photo.generic and places[0].photo.caption == "Miami"


def test_a_city_photo_is_labelled_as_general_in_the_plan() -> None:
    stand_in = Photo(url="https://upload.wikimedia.org/m.jpg", page="p", credit="c",
                     caption="Miami", generic=True)
    state = {
        "params": TripParams(destination="Miami", duration_days=2),
        "research": ResearchOutput(destination="Miami", attractions=[
            Attraction(name="Lunch on Washington Avenue", photo=stand_in),
        ]),
    }
    plan = aggregator(state, LLM(provider="mock"))["final_plan"]
    assert "general photo of Miami, not this place" in plan
    assert "![Miami (general photo)]" in plan
