"""Adapters must always answer, with or without an API key."""

from __future__ import annotations

from backend.adapters.base import Adapter
from backend.adapters.weather import WeatherAdapter


def test_weather_falls_back_to_mock_without_a_key() -> None:
    result = WeatherAdapter(api_key=None).fetch(destination="Kyoto")

    assert result.is_mock
    assert result.provider == "openweather"
    assert result.data["avg_high_c"] > result.data["avg_low_c"]
    assert result.data["advice"]


def test_unknown_destination_still_returns_something_usable() -> None:
    result = WeatherAdapter(api_key=None).fetch(destination="Somewhereville")

    assert result.is_mock
    assert result.data["summary"]


def test_live_failure_degrades_to_mock_instead_of_raising() -> None:
    class Boom(Adapter):
        name = "boom"

        def _fetch_live(self, **kwargs):
            raise RuntimeError("network down")

        def _fetch_mock(self, **kwargs):
            return {"ok": True}

    result = Boom(api_key="present").fetch()

    assert result.is_mock
    assert result.data == {"ok": True}
