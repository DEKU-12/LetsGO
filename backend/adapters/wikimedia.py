"""Photos of verified places, from Wikidata and Wikimedia Commons.

The photo is tied to the place the map verified, not found by searching a name:
the Geoapify entry that confirmed a place carries its Wikidata id (see
``PlacesAdapter``), Wikidata's "image" property (P18) names a Commons file, and
Commons supplies the thumbnail, author and licence. So "Gion" cannot come back
with a photo of a different Gion.

When a map entry has no Wikidata link, a Wikipedia search is the fallback,
guarded by title *and* location (see ``_by_search``).

No API key. Wikimedia refuses clients that do not identify themselves with a
contact, so requests carry the project URL.
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from html import unescape
from typing import Any

import httpx

from backend.adapters.base import Adapter
from backend.adapters.places import distance_km, names_match

USER_AGENT = "LetsGO/0.1 (https://github.com/DEKU-12/LetsGO)"

#: Thumbnail width in pixels: sharp at the size the plan shows, small to load.
THUMB_WIDTH = 480

#: How far a Wikipedia article's coordinates may be from where the map put the
#: place. Generous because a large park's article sits at its centre (Shark
#: Valley is ~50 km from the Everglades' midpoint); the title must match too.
MAX_KM = 60.0

#: Only images from Wikimedia's own file servers reach the page.
_TRUSTED = ("https://upload.wikimedia.org/", "https://thumb.wikimedia.org/")


def _plain(html: str, limit: int = 60) -> str:
    """Commons credits arrive as HTML; keep the words."""
    text = unescape(re.sub(r"<[^>]+>", "", html or "")).strip()
    text = re.sub(r"\s+", " ", text)
    return text if len(text) <= limit else text[: limit - 1] + "…"


class WikimediaAdapter(Adapter):
    """Wikidata ids in, photos with credits out."""

    name = "wikimedia"

    @property
    def live(self) -> bool:
        return True  # public APIs, no key; ids only exist when the map is live

    def _fetch_live(
        self, *, qids: list[str] = (), searches: list[dict[str, Any]] = (), **_: Any
    ) -> dict[str, Any]:
        """Photos keyed by Wikidata id (``qids``), or by search key (``searches``)."""
        headers = {"User-Agent": USER_AGENT}
        with httpx.Client(timeout=15.0, headers=headers, follow_redirects=True) as client:
            files = self._by_search(client, searches) if searches else self._by_qid(client, qids)
            return self._commons(client, files) if files else {}

    def _by_qid(self, client: httpx.Client, qids: list[str]) -> dict[str, tuple[str, str]]:
        """Wikidata id -> (Commons file, label), from each item's "image" (P18)."""
        qids = list(dict.fromkeys(q for q in qids if re.fullmatch(r"Q\d+", q or "")))
        entities: dict[str, Any] = {}
        for i in range(0, len(qids), 50):  # the API's batch limit
            response = client.get("https://www.wikidata.org/w/api.php", params={
                "action": "wbgetentities", "ids": "|".join(qids[i:i + 50]),
                "props": "claims|labels", "languages": "en", "format": "json",
            })
            response.raise_for_status()
            entities.update(response.json().get("entities", {}))

        files: dict[str, tuple[str, str]] = {}
        for qid, entity in entities.items():
            claims = (entity.get("claims") or {}).get("P18") or []
            file = (((claims[0] if claims else {}).get("mainsnak") or {})
                    .get("datavalue") or {}).get("value")
            if file:
                label = ((entity.get("labels") or {}).get("en") or {}).get("value", "")
                files[qid] = (file, label)
        return files

    def _by_search(
        self, client: httpx.Client, searches: list[dict[str, Any]]
    ) -> dict[str, tuple[str, str]]:
        """Search key -> (Commons file, article title), from Wikipedia.

        The fallback for places whose map entry has no Wikidata link. A search
        can return a different place with a similar name, so a result counts
        only if its title matches one of the place's names *and* the article's
        own coordinates are within ``MAX_KM`` of where the map put the place.
        Each search is ``{"key", "query", "names", "lat", "lon"}``.
        """
        def one(item: dict[str, Any]) -> tuple[str, tuple[str, str] | None]:
            response = client.get("https://en.wikipedia.org/w/api.php", params={
                "action": "query", "format": "json", "generator": "search",
                "gsrsearch": item["query"], "gsrlimit": 3, "gsrnamespace": 0,
                "prop": "pageprops|coordinates", "ppprop": "page_image_free",
            })
            response.raise_for_status()
            pages = (response.json().get("query") or {}).get("pages") or {}
            for page in sorted(pages.values(), key=lambda pg: pg.get("index", 99)):
                title = page.get("title", "")
                file = (page.get("pageprops") or {}).get("page_image_free")
                where = (page.get("coordinates") or [{}])[0]
                if not file or "lat" not in where or "lon" not in where:
                    continue
                if not any(names_match(n, title) for n in item["names"]):
                    continue
                if distance_km(item["lat"], item["lon"], where["lat"], where["lon"]) > MAX_KM:
                    continue
                return item["key"], (file, title)
            return item["key"], None

        with ThreadPoolExecutor(max_workers=4) as pool:
            return {k: v for k, v in pool.map(one, searches) if v}

    def _commons(
        self, client: httpx.Client, files: dict[str, tuple[str, str]]
    ) -> dict[str, Any]:
        """(Commons file, label) per key -> thumbnail, page, credit, label."""
        response = client.get("https://commons.wikimedia.org/w/api.php", params={
            "action": "query", "prop": "imageinfo", "format": "json",
            "iiprop": "url|extmetadata", "iiurlwidth": THUMB_WIDTH,
            "titles": "|".join(f"File:{f}" for f, _ in files.values()),
        })
        response.raise_for_status()
        query = response.json().get("query", {})

        # The API normalises titles (underscores, capitals); map them back.
        renamed = {n["from"]: n["to"] for n in query.get("normalized", [])}
        info_by_title = {
            page.get("title"): (page.get("imageinfo") or [{}])[0]
            for page in (query.get("pages") or {}).values()
        }

        photos: dict[str, Any] = {}
        for key, (file, label) in files.items():
            title = renamed.get(f"File:{file}", f"File:{file}")
            info = info_by_title.get(title) or {}
            # Drop the utm_* tracking query Commons appends; the path alone serves.
            thumb = (info.get("thumburl") or "").split("?", 1)[0]
            if not thumb.startswith(_TRUSTED):
                continue
            meta = info.get("extmetadata") or {}
            artist = _plain((meta.get("Artist") or {}).get("value", ""))
            licence = _plain((meta.get("LicenseShortName") or {}).get("value", ""), 30)
            photos[key] = {
                "url": thumb,
                "page": info.get("descriptionurl") or f"https://commons.wikimedia.org/wiki/{title}",
                "credit": ", ".join(x for x in (artist, licence) if x),
                "label": label,
            }
        return photos

    def _fetch_mock(self, **_: Any) -> dict[str, Any]:
        return {}  # no photos rather than fake ones
