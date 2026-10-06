"""How often do places get a photo of themselves, and readable opening hours?

    uv run python -m eval.photos

No model involved: a fixed list of real places is run through the same map
verification, place details and Wikimedia lookups a plan uses. Reports:

* **photo** — a real photo of the place or something linked to it, from the
  map's Wikidata link or the guarded Wikipedia fallback.
* **of the place** — and its title matches the place (no caption needed). A
  photo of something related (a café's famous pastry) is shown with a caption,
  so it is honest, but it is not a photo of the place.
* **city stand-in** — neither worked, so the destination's photo is shown,
  labelled as a general photo. Every place gets one of the three.
* **hours read** — opening hours exist and ``backend/hours.py`` could read them
  for certain; only those are used by the schedule check.

Results go to `eval_results/photos-<timestamp>.json`.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from backend.adapters.places import PlacesAdapter
from backend.agents.destination_research import attach_photos
from backend.config import PROJECT_ROOT
from backend.hours import parse
from backend.state import Attraction

RESULTS_DIR = PROJECT_ROOT / "eval_results"

#: Famous landmarks, a covered market, a café, a neighbourhood, a viewpoint and
#: a park — the mix a real plan contains, not only easy cases.
PLACES: dict[str, tuple[str, ...]] = {
    "Kyoto": ("Kinkaku-ji", "Fushimi Inari Taisha", "Nishiki Market", "Gion",
              "Philosopher's Path"),
    "Rome": ("Colosseum", "Pantheon", "Trastevere", "Vatican Museums", "Villa Borghese"),
    "Lisbon": ("Pastéis de Belém", "Time Out Market Lisboa", "Belém Tower",
               "Miradouro da Senhora do Monte", "Alfama"),
    # Names as a real plan wrote them, combined ones included.
    "Miami": ("South Beach and Ocean Drive", "Little Havana and Calle Ocho", "Wynwood Walls",
              "Vizcaya Museum and Gardens", "Pérez Art Museum Miami",
              "Everglades National Park – Shark Valley"),
}


def main() -> int:
    started = datetime.now(timezone.utc)
    places = PlacesAdapter()
    if not places.live:
        print("GEOAPIFY_API_KEY is not set — there is nothing to look up.")
        return 1

    rows = []
    print(f"  {'place':<38}{'photo':<8}{'of it':<8}{'stand-in':<10}{'hours':<7}detail")
    for city, names in PLACES.items():
        found = places.fetch(destination=city, names=list(names), details=True).data
        details = found.get("details") or {}
        attractions = []
        for name in names:
            lat, lon = (found.get("coords") or {}).get(name, (None, None))
            attractions.append(Attraction(
                name=name, lat=lat, lon=lon,
                opening_hours=(details.get(name) or {}).get("opening_hours"),
            ))
        attach_photos(attractions, city, details, found.get("centre"), set())

        for a in attractions:
            photo = a.photo
            real = photo is not None and not photo.generic
            row = {
                "place": f"{a.name}, {city}",
                "verified": found["confirmed"].get(a.name) is not None,
                "photo": real,
                "of_place": real and photo.caption is None,
                "stand_in": photo is not None and photo.generic,
                "caption": photo.caption if real else None,
                "hours": a.opening_hours,
                "hours_read": parse(a.opening_hours) is not None,
            }
            rows.append(row)
            mark = lambda v: "yes" if v else "-"  # noqa: E731
            detail = (f"shows {row['caption']}" if row["caption"] else "") + \
                ("" if row["verified"] else "  (not verified on the map)")
            print(f"  {row['place'][:37]:<38}{mark(row['photo']):<8}{mark(row['of_place']):<8}"
                  f"{mark(row['stand_in']):<10}{mark(row['hours_read']):<7}{detail[:60]}")

    n = len(rows)
    print(f"\n  real photo            {sum(r['photo'] for r in rows)}/{n}")
    print(f"  photo of the place    {sum(r['of_place'] for r in rows)}/{n}")
    print(f"  city stand-in         {sum(r['stand_in'] for r in rows)}/{n}")
    print(f"  no photo at all       {sum(not r['photo'] and not r['stand_in'] for r in rows)}/{n}")
    print(f"  hours read for sure   {sum(r['hours_read'] for r in rows)}/{n}")

    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"photos-{started:%Y%m%dT%H%M%SZ}.json"
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    print(f"\n  saved {path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
