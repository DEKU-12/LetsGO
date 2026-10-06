"""Geographic verification of recommended places.

Geoapify is used here as a *verifier*, not a source. That is a deliberate
inversion of the obvious design, and the reason is measured rather than assumed:
a radius search around Kyoto returns commemorative plaques and the city hall,
while Fushimi Inari and Kinkaku-ji do not appear at all. OpenStreetMap is a
database of what exists near a point, not a ranking of what is worth seeing.
Using it to pick attractions would make plans measurably worse.

So the model proposes — it is good at this — and Geoapify confirms each name
exists at that destination.

**"Unverified" means "could not confirm", never "does not exist."** The matcher
is tuned to produce no false positives at the cost of some false negatives: on a
12-place benchmark it confirmed 6 of 8 real places and rejected all 4 invented
ones. The two misses were real places whose OpenStreetMap name is in the local
language (Torre de Belém, Museu Nacional do Azulejo). Nothing user-facing calls
an unverified place fake; the number is reported as a confirmation rate.
"""

from __future__ import annotations

import difflib
import re
import unicodedata
from typing import Any

import httpx

from backend.adapters.base import Adapter
from backend.config import settings

#: How far from the city centre a place may be and still count as "there".
SEARCH_RADIUS_M = 60_000

#: Candidates inspected per name. The first hit is often a café with a similar
#: name; the real place is usually within the first few.
CANDIDATES = 5

#: Similarity thresholds, chosen from a labelled benchmark rather than by feel.
SEQUENCE_THRESHOLD = 0.85
TOKEN_THRESHOLD = 0.5

#: Words too common in place names to carry identity.
#: Includes the common non-English forms, because a place named in two
#: languages otherwise looks like two different places sharing a generic word.
_GENERIC = {
    # articles and connectives
    "the", "of", "de", "do", "da", "del", "della", "la", "le", "el", "los",
    "and", "e", "y", "et", "di", "dos", "das", "a", "au", "aux",
    # English place words
    "museum", "castle", "tower", "market", "monastery", "temple", "palace",
    "national", "park", "garden", "gardens", "shrine", "cathedral", "church",
    "gallery", "district", "quarter", "bridge", "square", "old", "new",
    "great", "grand", "royal", "city", "centre", "center", "house", "hall",
    # the same words in the languages these names usually arrive in
    "museu", "museo", "musee", "musée", "museum",
    "castelo", "castillo", "castello", "chateau", "château",
    "torre", "torres", "palacio", "palácio", "palazzo", "palais",
    "igreja", "iglesia", "chiesa", "eglise", "église", "duomo", "basilica",
    "jardim", "jardin", "giardino", "parque", "parc",
    "praça", "praca", "plaza", "piazza", "place", "platz",
    "mercado", "mercato", "marche", "marché", "nacional", "nazionale",
    "-ji", "-dera", "jinja", "taisha",
}


def _tokens(value: str) -> list[str]:
    folded = unicodedata.normalize("NFKD", value.lower())
    folded = "".join(c for c in folded if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9 ]", " ", folded).split()


def name_variants(name: str) -> list[str]:
    """Forms of a name worth matching against.

    Models often write "Kinkaku-ji (Golden Pavilion)" or "National Tile Museum
    (Museu Nacional do Azulejo)". Both halves are real names for the place, and
    OpenStreetMap will usually carry one of them, so try each.
    """
    variants = [name.strip()]
    stripped = re.sub(r"\s*\([^)]*\)", "", name).strip()
    if stripped and stripped != name.strip():
        variants.append(stripped)
    for inner in re.findall(r"\(([^)]*)\)", name):
        inner = inner.strip()
        if len(inner) > 3:
            variants.append(inner)
    return variants


def names_match(queried: str, found: str) -> bool:
    """Is `found` plausibly the same place as `queried`?

    Two signals, either sufficient: overall string similarity, or overlap of the
    distinctive (non-generic) words. Both are needed because accents and word
    order defeat the first, and translated names defeat the second.
    """
    left, right = _tokens(queried), _tokens(found)
    if not left or not right:
        return False

    sequence = difflib.SequenceMatcher(None, " ".join(left), " ".join(right)).ratio()
    if sequence >= SEQUENCE_THRESHOLD:
        return True

    distinctive_left = set(left) - _GENERIC
    distinctive_right = set(right) - _GENERIC
    if not distinctive_left or not distinctive_right:
        return False

    overlap = len(distinctive_left & distinctive_right)
    union = len(distinctive_left | distinctive_right)
    return union > 0 and overlap / union >= TOKEN_THRESHOLD


class PlacesAdapter(Adapter):
    """Confirms that named places exist at a destination."""

    name = "geoapify"

    def __init__(self, api_key: str | None = None) -> None:
        super().__init__(api_key if api_key is not None else settings.geoapify_api_key)

    # -- live ---------------------------------------------------------------

    def _geocode(self, client: httpx.Client, destination: str) -> tuple[float, float]:
        response = client.get(
            "https://api.geoapify.com/v1/geocode/search",
            params={"text": destination, "limit": 1, "lang": "en", "apiKey": self.api_key},
        )
        response.raise_for_status()
        features = response.json().get("features") or []
        if not features:
            raise ValueError(f"Geoapify could not locate {destination!r}")
        properties = features[0]["properties"]
        return properties["lat"], properties["lon"]

    def _fetch_live(self, *, destination: str, names: list[str], **_: Any) -> dict[str, Any]:
        with httpx.Client(timeout=20.0) as client:
            lat, lon = self._geocode(client, destination)

            confirmed: dict[str, str | None] = {}
            coords: dict[str, tuple[float, float]] = {}
            for name in names:
                variants = name_variants(name)
                confirmed[name] = None

                # Search each form. A place written in English may only be
                # findable under its local name, and vice versa.
                for query in variants:
                    response = client.get(
                        "https://api.geoapify.com/v1/geocode/search",
                        params={
                            "text": query,
                            "limit": CANDIDATES,
                            "lang": "en",
                            "apiKey": self.api_key,
                            "filter": f"circle:{lon},{lat},{SEARCH_RADIUS_M}",
                            "bias": f"proximity:{lon},{lat}",
                        },
                    )
                    response.raise_for_status()

                    for feature in response.json().get("features") or []:
                        properties = feature["properties"]
                        candidate = (
                            properties.get("name") or properties.get("address_line1") or ""
                        )
                        if candidate and any(names_match(v, candidate) for v in variants):
                            confirmed[name] = candidate
                            if "lat" in properties and "lon" in properties:
                                coords[name] = (properties["lat"], properties["lon"])
                            break
                    if confirmed[name]:
                        break

        return {"confirmed": confirmed, "coords": coords, "centre": {"lat": lat, "lon": lon}}

    # -- mock ---------------------------------------------------------------

    def _fetch_mock(self, *, names: list[str], **_: Any) -> dict[str, Any]:
        """Confirm nothing.

        A mock verifier that returned "verified" would defeat the entire point
        of the check, so with no key every place is simply unchecked and the
        plan reports no confirmation rate at all.
        """
        return {"confirmed": {name: None for name in names}, "coords": {}, "centre": None}
