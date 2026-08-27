"""Accommodation options.

There is no live mode. Real hotel inventory APIs (Booking, Expedia, Amadeus)
are gated behind commercial agreements, so this adapter serves generated data
and says so: every result carries ``source="mock"``, and the aggregator prints
that provenance under the plan. Fake data presented as real would be the one
genuinely dishonest thing this project could do.

The generation is deterministic — the same destination and budget always give
the same shortlist — so eval runs are reproducible.
"""

from __future__ import annotations

import hashlib
from typing import Any

from backend.adapters.base import Adapter

#: Nightly USD target per stated budget level, before the city multiplier.
NIGHTLY_BASELINE: dict[str, float] = {
    "budget": 45.0,
    "mid-range": 130.0,
    "luxury": 380.0,
}

#: Rough cost-of-stay multipliers. Everywhere else is 1.0.
_CITY_MULTIPLIER: dict[str, float] = {
    "tokyo": 1.15, "kyoto": 1.1, "japan": 1.1,
    "paris": 1.3, "london": 1.4, "new york": 1.5, "reykjavik": 1.45,
    "iceland": 1.45, "norway": 1.4, "switzerland": 1.6,
    "rome": 1.1, "italy": 1.05, "barcelona": 1.0, "spain": 0.95,
    "lisbon": 0.85, "portugal": 0.85, "greece": 0.9,
    "bangkok": 0.5, "thailand": 0.5, "vietnam": 0.45, "peru": 0.55,
    "marrakesh": 0.6, "morocco": 0.6,
}

#: Neighbourhoods used to make options feel located rather than generic.
_AREAS: dict[str, tuple[str, ...]] = {
    "tokyo": ("Shinjuku", "Asakusa", "Shibuya", "Yanaka"),
    "kyoto": ("Gion", "Higashiyama", "Karasuma", "Arashiyama"),
    "japan": ("Shinjuku, Tokyo", "Gion, Kyoto", "Namba, Osaka", "Asakusa, Tokyo"),
    "paris": ("Le Marais", "Latin Quarter", "Canal Saint-Martin", "Montmartre"),
    "rome": ("Trastevere", "Monti", "Prati", "Testaccio"),
    "italy": ("Trastevere, Rome", "Oltrarno, Florence", "Cannaregio, Venice", "Monti, Rome"),
    "lisbon": ("Alfama", "Bairro Alto", "Principe Real", "Belem"),
    "barcelona": ("Gracia", "El Born", "Eixample", "Poble Sec"),
    "london": ("Shoreditch", "Bloomsbury", "South Bank", "Camden"),
    "bangkok": ("Riverside", "Sukhumvit", "Old Town", "Silom"),
}

_GENERIC_AREAS = ("Old Town", "City Centre", "Riverside", "Station Quarter")

_STYLES: tuple[tuple[str, str, float], ...] = (
    # (name template, why, price factor)
    ("The {area} Guesthouse", "Small, well reviewed, walkable to the centre.", 0.75),
    ("Hotel {area}", "Reliable mid-size hotel with a good breakfast.", 1.0),
    ("{area} Boutique Rooms", "Quieter side street, more character than chain hotels.", 1.2),
    ("Casa {area}", "Apartment-style with a kitchenette; good for longer stays.", 0.9),
    ("{area} Grand", "Larger property with a spa and a proper bar.", 1.6),
)


def _multiplier(destination: str) -> float:
    key = destination.strip().lower()
    return next((v for k, v in _CITY_MULTIPLIER.items() if k in key), 1.0)


def _areas(destination: str) -> tuple[str, ...]:
    key = destination.strip().lower()
    return next((v for k, v in _AREAS.items() if k in key), _GENERIC_AREAS)


def _seed(destination: str, budget: str) -> int:
    digest = hashlib.sha256(f"{destination}|{budget}".encode()).hexdigest()
    return int(digest[:8], 16)


class LodgingAdapter(Adapter):
    """Mock-only. Real hotel inventory needs a commercial agreement."""

    name = "lodging"

    def __init__(self, api_key: str | None = None) -> None:
        super().__init__(api_key)

    def _fetch_live(self, **_: Any) -> Any:
        raise NotImplementedError(
            "no live hotel provider is configured; real inventory APIs are "
            "gated behind commercial agreements"
        )

    def _fetch_mock(
        self, *, destination: str, budget: str | None = None, nights: int = 1, **_: Any
    ) -> dict[str, Any]:
        level = budget if budget in NIGHTLY_BASELINE else "mid-range"
        target = NIGHTLY_BASELINE[level] * _multiplier(destination)

        areas = _areas(destination)
        seed = _seed(destination, level)

        options = []
        for index, (template, why, factor) in enumerate(_STYLES):
            area = areas[(seed + index) % len(areas)]
            price = round(target * factor, -1) or 10.0
            # Deterministic 3.9-4.8 spread, better ratings for pricier rooms.
            rating = round(3.9 + ((seed >> index) % 6) / 10 + (0.1 if factor > 1 else 0), 1)
            options.append(
                {
                    "name": template.format(area=area.split(",")[0]),
                    "area": area,
                    "price_per_night_usd": price,
                    "rating": min(rating, 4.9),
                    "why": why,
                }
            )

        return {
            "options": options,
            "nightly_budget_usd": round(target, -1),
            "nights": nights,
        }
