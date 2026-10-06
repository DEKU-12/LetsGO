"""Remembering a traveller's lasting preferences.

After a plan, one small model call reads the request and pulls out what would
still be true on *any* trip — "I'm vegetarian", "we travel with two kids", "no
early mornings" — as opposed to what is about this trip only — "it's for my
mum's birthday", "on a tight budget this time".

Nothing found here is saved on its own. The API returns these as suggestions
and the traveller chooses which to keep; saving personal details someone
mentioned in passing, without asking, is the failure this design avoids.
Saved preferences then reach every planning agent through ``profile_note``.
"""

from __future__ import annotations

from typing import Any

from backend.llm import LLM, register_mock

#: Upper bounds on what one traveller can store, so a profile stays a short
#: list a person can read, not a dossier.
MAX_PREFERENCES = 12
MAX_LENGTH = 80

EXTRACT_SYSTEM = """You read a travel request and find the traveller's LASTING preferences:
things that would still be true on any other trip they take.

Return JSON: {"lasting": [str]}

Lasting (include): diet and allergies, mobility or accessibility needs, who they
always travel with (kids, a partner, a dog), pace and timing habits ("hates early
mornings", "max two sights a day"), things they always avoid.

Not lasting (leave out): this trip's destination, dates, length, budget, purpose
or occasion (a birthday, a honeymoon, a business meeting), and one-off interests
for this trip.

Rules:
- Short lowercase phrases in the traveller's terms, e.g. "vegetarian",
  "travels with young kids", "no early mornings".
- Only what they actually said. Do not infer ("loves food" does not mean
  "foodie"). An empty list is the usual answer."""

#: Keyword stand-in for the model when no API key is set.
_MOCK_RULES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("vegetarian",), "vegetarian"),
    (("vegan",), "vegan"),
    (("gluten",), "gluten-free"),
    (("kids", "children"), "travels with kids"),
    (("wheelchair",), "uses a wheelchair"),
    (("early morning", "early start"), "no early mornings"),
)


@register_mock("extract_preferences")
def _mock_extract(context: dict[str, Any]) -> dict[str, Any]:
    text = str(context.get("request", "")).lower()
    return {"lasting": [label for words, label in _MOCK_RULES if any(w in text for w in words)]}


def clean(preferences: list[Any]) -> list[str]:
    """Trim, lowercase, drop blanks and duplicates, and cap the list."""
    seen: dict[str, None] = {}
    for item in preferences:
        text = str(item).strip().lower()[:MAX_LENGTH]
        if text:
            seen.setdefault(text, None)
    return list(seen)[:MAX_PREFERENCES]


def suggest_preferences(request: str, llm: LLM, known: list[str]) -> list[str]:
    """Lasting preferences in `request` that are not saved yet.

    Raises LLMError on a provider failure; the caller decides whether a missing
    suggestion matters (for a finished plan, it does not).
    """
    raw = llm.json(
        task="extract_preferences",
        system=EXTRACT_SYSTEM,
        prompt=f"Request: {request}",
        context={"request": request},
        max_tokens=512,
    )
    have = {k.lower() for k in known}
    return [p for p in clean(raw.get("lasting") or []) if p not in have]
