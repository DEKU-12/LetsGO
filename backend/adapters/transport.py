"""Flights, rail and local transport.

Mock-only, for the same reason as lodging: real flight inventory (Amadeus,
Sabre, airline APIs) needs a commercial agreement, and the free tiers that do
exist return cached fares too stale to plan against. Results are tagged
``source="mock"`` and the plan says so.

Local transport is the part that stays broadly true regardless — a city's
metro, its transit pass, whether taxis are worth it — so that is where the
detail sits.
"""

from __future__ import annotations

from typing import Any

from backend.adapters.base import Adapter

#: (airport, typical return economy fare band in USD) by destination keyword.
_GATEWAY: dict[str, tuple[str, int]] = {
    "japan": ("Tokyo Haneda (HND) or Narita (NRT)", 900),
    "tokyo": ("Tokyo Haneda (HND)", 900),
    "kyoto": ("Osaka Kansai (KIX), then rail to Kyoto", 900),
    "thailand": ("Bangkok Suvarnabhumi (BKK)", 800),
    "bangkok": ("Bangkok Suvarnabhumi (BKK)", 800),
    "vietnam": ("Hanoi (HAN) or Ho Chi Minh City (SGN)", 800),
    "italy": ("Rome Fiumicino (FCO)", 550),
    "rome": ("Rome Fiumicino (FCO)", 550),
    "france": ("Paris Charles de Gaulle (CDG)", 500),
    "paris": ("Paris Charles de Gaulle (CDG)", 500),
    "spain": ("Madrid (MAD) or Barcelona (BCN)", 500),
    "barcelona": ("Barcelona El Prat (BCN)", 500),
    "portugal": ("Lisbon Humberto Delgado (LIS)", 500),
    "lisbon": ("Lisbon Humberto Delgado (LIS)", 500),
    "london": ("London Heathrow (LHR)", 450),
    "iceland": ("Reykjavik Keflavik (KEF)", 600),
    "reykjavik": ("Reykjavik Keflavik (KEF)", 600),
    "morocco": ("Marrakesh Menara (RAK)", 550),
    "marrakesh": ("Marrakesh Menara (RAK)", 550),
    "greece": ("Athens (ATH)", 550),
    "peru": ("Lima (LIM)", 850),
    "new york": ("New York JFK or Newark (EWR)", 600),
}

#: Local transport worth knowing about, by destination keyword.
_LOCAL: dict[str, tuple[tuple[str, str, float | None], ...]] = {
    "japan": (
        ("Rail pass", "7-day JR Pass covers the Shinkansen between Tokyo and Kyoto.", 350.0),
        ("IC card", "Suica or Pasmo for metro, buses and convenience stores.", 20.0),
        ("Walking", "Both cities reward walking; most districts are compact.", None),
    ),
    "italy": (
        ("High-speed rail", "Frecciarossa between major cities; book ahead for cheaper fares.", 45.0),
        ("Metro and bus", "72-hour city pass covers most of what you need.", 20.0),
        ("Walking", "Historic centres are small and largely pedestrian.", None),
    ),
    "france": (
        ("Metro", "Carnet or Navigo Easy card; the metro reaches almost everything.", 25.0),
        ("TGV", "High-speed rail if you leave the city.", 60.0),
        ("Walking", "Central arrondissements are walkable end to end.", None),
    ),
    "portugal": (
        ("Metro and tram", "Viva Viagem card; tram 28 doubles as a sightseeing route.", 15.0),
        ("Regional rail", "Cheap and frequent to Sintra and Cascais.", 10.0),
        ("Walking", "Steep — comfortable shoes matter more than usual.", None),
    ),
    "thailand": (
        ("BTS Skytrain", "Rabbit card; fastest way across the city in traffic.", 15.0),
        ("River boat", "Chao Phraya express boats reach many temples.", 5.0),
        ("Grab", "Ride hailing is cheap and avoids taxi meter disputes.", 25.0),
    ),
}

_LOCAL_DEFAULT: tuple[tuple[str, str, float | None], ...] = (
    ("Public transit pass", "Multi-day pass is usually cheaper than single tickets.", 25.0),
    ("Ride hailing", "Useful for late arrivals and early departures.", 30.0),
    ("Walking", "Most of the centre is comfortably walkable.", None),
)


def _lookup(destination: str, table: dict[str, Any], default: Any) -> Any:
    key = destination.strip().lower()
    return next((v for k, v in table.items() if k in key), default)


class TransportAdapter(Adapter):
    """Mock-only. Real fares need a commercial flight data agreement."""

    name = "transport"

    def _fetch_live(self, **_: Any) -> Any:
        raise NotImplementedError(
            "no live flight provider is configured; fare APIs are gated behind "
            "commercial agreements"
        )

    def _fetch_mock(
        self, *, destination: str, travelers: int = 1, **_: Any
    ) -> dict[str, Any]:
        airport, fare = _lookup(destination, _GATEWAY, ("the nearest international airport", 600))
        local = _lookup(destination, _LOCAL, _LOCAL_DEFAULT)

        return {
            "inbound": [
                {
                    "mode": "Flight",
                    "description": f"Return economy into {airport}.",
                    "est_cost_usd": float(fare * travelers),
                },
                {
                    "mode": "Airport transfer",
                    "description": "Express train or bus into the centre; faster than a taxi at peak.",
                    "est_cost_usd": 15.0 * travelers,
                },
            ],
            "local": [
                {"mode": mode, "description": description, "est_cost_usd": cost}
                for mode, description, cost in local
            ],
        }
