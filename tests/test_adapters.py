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


# -- place verification -----------------------------------------------------


def test_name_matching_accepts_real_variants_and_rejects_near_misses() -> None:
    """Thresholds come from a labelled benchmark; this pins the behaviour."""
    from backend.adapters.places import names_match

    # Same place, spelled or ordered differently.
    assert names_match("Jeronimos Monastery", "Jerónimos Monastery")
    assert names_match("Castelo de Sao Jorge", "São Jorge Castle")
    assert names_match("Nijo Castle", "Nijō Castle")
    assert names_match("Fushimi Inari Taisha", "Fushimi Inari-taisha")

    # Different places that a naive matcher would confuse.
    assert not names_match("Palacio do Nada", "Palácio do Mitelo")
    assert not names_match("Temple of the Silver Fox", "Entry of Theotokos into the Temple")
    assert not names_match("Belem Tower", 'Belém 2a8 "Bistrô"')


def test_generic_words_alone_never_confirm_a_match() -> None:
    """'Museum' matching 'Museum' must not count as identifying a place."""
    from backend.adapters.places import names_match

    assert not names_match("National Museum", "City Museum")
    assert not names_match("Old Town Park", "Riverside Park")


def test_places_mock_confirms_nothing() -> None:
    """A mock verifier that verified things would defeat the point of the check."""
    from backend.adapters.places import PlacesAdapter

    result = PlacesAdapter(api_key=NO_KEY).fetch(
        destination="Lisbon", names=["Jeronimos Monastery", "Museum of Imaginary Tiles"]
    )
    assert result.is_mock
    assert all(v is None for v in result.data["confirmed"].values())
