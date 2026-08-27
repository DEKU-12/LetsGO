"""Weather outlook for a destination.

Live source: OpenWeather (current weather + 5 day forecast endpoints).
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


class WeatherAdapter(Adapter):
    name = "openweather"

    def __init__(self, api_key: str | None = None) -> None:
        super().__init__(api_key if api_key is not None else settings.openweather_api_key)

    def _fetch_live(self, *, destination: str, **_: Any) -> dict[str, Any]:
        with httpx.Client(timeout=10.0) as client:
            geo = client.get(
                "https://api.openweathermap.org/geo/1.0/direct",
                params={"q": destination, "limit": 1, "appid": self.api_key},
            )
            geo.raise_for_status()
            places = geo.json()
            if not places:
                raise ValueError(f"OpenWeather could not geocode {destination!r}")
            lat, lon = places[0]["lat"], places[0]["lon"]

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

    def _fetch_mock(self, *, destination: str, **_: Any) -> dict[str, Any]:
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
