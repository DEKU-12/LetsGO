"""Weather outlook for a destination.

Live source: OpenWeather (current weather + 5 day forecast endpoints).
``fetch(destination=...)`` returns the trip outlook; ``fetch(destination=...,
when="now")`` returns conditions right now, for trip-day help.
Mock source: a small climate table keyed by city, with a generic temperate
default. Mock values are plausible, not real — anything built on them is
labelled as mock further up the stack.
"""

from __future__ import annotations

from typing import Any

import httpx

from backend.adapters.base import Adapter
from backend.config import settings

# Rough seasonal averages (avg high C, avg low C, one-line summary).
_CLIMATE: dict[str, tuple[float, float, str]] = {
    "tokyo": (21.0, 13.0, "Mild with occasional rain; humid in summer."),
    "kyoto": (22.0, 12.0, "Mild days, cooler evenings; wet in June."),
    "osaka": (22.0, 14.0, "Warm and humid; light rain likely."),
    "paris": (17.0, 9.0, "Cool and changeable with intermittent showers."),
    "rome": (23.0, 13.0, "Warm and mostly dry; hot at midday in summer."),
    "barcelona": (23.0, 15.0, "Warm, sunny, sea breeze in the afternoons."),
    "london": (15.0, 8.0, "Cool, grey, frequent light rain."),
    "reykjavik": (10.0, 3.0, "Cold, windy, fast-changing conditions."),
    "bangkok": (33.0, 26.0, "Hot and humid with afternoon thunderstorms."),
    "lisbon": (21.0, 13.0, "Mild and sunny with an Atlantic breeze."),
    "new york": (18.0, 10.0, "Four distinct seasons; variable shoulder months."),
    "marrakesh": (28.0, 14.0, "Hot dry days, sharply cooler nights."),
    # Country-level fallbacks, checked after the cities above.
    "japan": (21.0, 13.0, "Mild with clear shoulder seasons; humid in midsummer."),
    "italy": (23.0, 13.0, "Warm and mostly dry; hot inland at midday."),
    "france": (18.0, 10.0, "Cool to mild and changeable; wetter in the north."),
    "iceland": (10.0, 3.0, "Cold, windy, fast-changing conditions."),
    "thailand": (33.0, 26.0, "Hot and humid with a distinct rainy season."),
    "portugal": (21.0, 13.0, "Mild and sunny with an Atlantic breeze."),
    "spain": (24.0, 14.0, "Warm and sunny; very hot inland in summer."),
}

_DEFAULT = (20.0, 11.0, "Temperate conditions; pack layers.")


def _advice(high: float, low: float) -> str:
    if high >= 30:
        return "Plan indoor or shaded activities midday and carry water."
    if high <= 12:
        return "Pack a warm layer and a windproof outer shell."
    if high - low >= 12:
        return "Large day-night swing — layers are worth the packing space."
    return "Comfortable for walking; a light rain layer is sensible."


#: OpenWeather condition groups that make outdoor sightseeing miserable.
_BAD_SKIES = {"Rain", "Drizzle", "Thunderstorm", "Snow"}

#: Temperatures (C) outside which a day outdoors needs rethinking.
TOO_HOT_C = 34.0
TOO_COLD_C = 0.0


def bad_for_outdoors(main: str, temp_c: float | None) -> bool:
    """Would a sensible person move today's outdoor plans indoors?"""
    if main in _BAD_SKIES:
        return True
    return temp_c is not None and not (TOO_COLD_C < temp_c < TOO_HOT_C)


class WeatherAdapter(Adapter):
    name = "openweather"

    def __init__(self, api_key: str | None = None) -> None:
        super().__init__(api_key if api_key is not None else settings.openweather_api_key)

    def _locate(self, client: httpx.Client, destination: str) -> tuple[float, float]:
        geo = client.get(
            "https://api.openweathermap.org/geo/1.0/direct",
            params={"q": destination, "limit": 1, "appid": self.api_key},
        )
        geo.raise_for_status()
        places = geo.json()
        if not places:
            raise ValueError(f"OpenWeather could not geocode {destination!r}")
        return places[0]["lat"], places[0]["lon"]

    def _fetch_live(self, *, destination: str, when: str = "outlook", **_: Any) -> dict[str, Any]:
        with httpx.Client(timeout=10.0) as client:
            lat, lon = self._locate(client, destination)

            if when == "now":
                current = client.get(
                    "https://api.openweathermap.org/data/2.5/weather",
                    params={"lat": lat, "lon": lon, "units": "metric", "appid": self.api_key},
                )
                current.raise_for_status()
                body = current.json()
                main = body["weather"][0]["main"]
                temp = round(body["main"]["temp"], 1)
                return {
                    "description": body["weather"][0]["description"],
                    "temp_c": temp,
                    "bad": bad_for_outdoors(main, temp),
                }

            forecast = client.get(
                "https://api.openweathermap.org/data/2.5/forecast",
                params={"lat": lat, "lon": lon, "units": "metric", "appid": self.api_key},
            )
            forecast.raise_for_status()
            entries = forecast.json()["list"]

        highs = [e["main"]["temp_max"] for e in entries]
        lows = [e["main"]["temp_min"] for e in entries]
        conditions = [e["weather"][0]["description"] for e in entries]
        avg_high = round(sum(highs) / len(highs), 1)
        avg_low = round(sum(lows) / len(lows), 1)
        dominant = max(set(conditions), key=conditions.count)

        return {
            "summary": f"Forecast trending toward {dominant}.",
            "avg_high_c": avg_high,
            "avg_low_c": avg_low,
            "advice": _advice(avg_high, avg_low),
        }

    def _fetch_mock(self, *, destination: str, when: str = "outlook", **_: Any) -> dict[str, Any]:
        if when == "now":
            # Live conditions cannot be faked usefully: a mock that said "rain"
            # would rearrange real plans for weather that is not happening.
            return {"description": None, "temp_c": None, "bad": False}

        key = destination.strip().lower()
        high, low, summary = next(
            (v for k, v in _CLIMATE.items() if k in key),
            _DEFAULT,
        )
        return {
            "summary": summary,
            "avg_high_c": high,
            "avg_low_c": low,
            "advice": _advice(high, low),
        }
