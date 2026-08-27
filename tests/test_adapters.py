"""Adapters must always answer, with or without an API key.

`api_key=None` means "use whatever is configured", so it is not a way to test
the no-key path — on a machine with a real key in .env these tests would quietly
start making live calls. `api_key=""` is the explicit "no credential" case.
"""

from __future__ import annotations

from backend.adapters.base import Adapter
from backend.adapters.weather import WeatherAdapter

NO_KEY = ""


def test_weather_falls_back_to_mock_without_a_key() -> None:
    result = WeatherAdapter(api_key=NO_KEY).fetch(destination="Kyoto")

    assert result.is_mock
    assert result.provider == "openweather"
    assert result.data["avg_high_c"] > result.data["avg_low_c"]
    assert result.data["advice"]


def test_unknown_destination_still_returns_something_usable() -> None:
    result = WeatherAdapter(api_key=NO_KEY).fetch(destination="Somewhereville")

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


def test_api_key_never_reaches_the_log(caplog) -> None:
    """These APIs take the key as a query param, so errors quote it back."""
    import logging

    secret = "super-secret-key-value"

    class Leaky(Adapter):
        name = "leaky"

        def _fetch_live(self, **kwargs):
            raise RuntimeError(f"401 for url https://example.com/x?appid={secret}")

        def _fetch_mock(self, **kwargs):
            return {"ok": True}

    with caplog.at_level(logging.WARNING):
        result = Leaky(api_key=secret).fetch()

    assert result.is_mock
    assert secret not in caplog.text
    assert "<redacted>" in caplog.text


def test_configured_key_is_used_when_none_is_passed() -> None:
    """`None` means "fall back to configuration" — the documented behaviour."""
    from backend.config import settings

    adapter = WeatherAdapter(api_key=None)
    assert adapter.api_key == settings.openweather_api_key
