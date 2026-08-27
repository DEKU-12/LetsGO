"""Accommodation agent — lodging inside the stated budget.

Candidates come from the lodging adapter (mock by necessity — see that module).
The model's job is selection and justification, not invention: it picks from
the shortlist and says why each one suits this traveller. Anything it names
that was not on the shortlist is dropped.
"""

from __future__ import annotations

from typing import Any

from backend.adapters.lodging import NIGHTLY_BASELINE, LodgingAdapter
from backend.llm import LLM, LLMError, register_mock
from backend.state import AccommodationOutput, LodgingOption, TravelState

#: How far above the nightly target a stay may go before it is dropped.
BUDGET_TOLERANCE = 1.25

ACCOMMODATION_SYSTEM = """You choose where a traveller should stay.

Return JSON:
  {"options": [{"name": str, "why": str}], "note": str}

Rules:
- Choose 3 options from the candidate list you are given. Use the names exactly.
- "why" is one sentence tying the choice to this traveller's stated interests,
  budget and trip length. Do not restate the price; it is shown separately.
- Prefer options at or below the nightly target. Include one above it only if
  it is clearly worth the difference, and say why in that sentence.
- "note" is one short line of practical advice about staying in this place."""


@register_mock("accommodation")
def _mock_accommodation(context: dict[str, Any]) -> dict[str, Any]:
    """Cheapest-three selection standing in for the model when no key is set."""
    candidates = sorted(
        context.get("candidates") or [],
        key=lambda c: c.get("price_per_night_usd", 0),
    )
    return {
        "options": [{"name": c["name"], "why": c.get("why", "")} for c in candidates[:3]],
        "note": "Book refundable rates; prices in this shortlist are illustrative.",
    }


def accommodation(state: TravelState, llm: LLM) -> dict[str, Any]:
    """Graph node: shortlist places to stay."""
    params = state["params"]
    assert params is not None

    nights = max((params.duration_days or 1) - 1, 1)
    result = LodgingAdapter().fetch(
        destination=params.destination,
        budget=params.budget,
        nights=nights,
    )
    candidates: list[dict[str, Any]] = result.data["options"]
    target: float = result.data["nightly_budget_usd"]

    by_name = {c["name"]: c for c in candidates}
    listing = "\n".join(
        f"- {c['name']} ({c['area']}, ${c['price_per_night_usd']:.0f}/night, "
        f"rated {c['rating']}): {c['why']}"
        for c in candidates
    )

    try:
        raw = llm.json(
            task="accommodation",
            system=ACCOMMODATION_SYSTEM,
            prompt=(
                f"Destination: {params.destination}\n"
                f"Budget level: {params.budget or 'unstated'} "
                f"(target ${target:.0f}/night)\n"
                f"Nights: {nights}   Travellers: {params.travelers}\n"
                f"Interests: {', '.join(params.preferences) or 'none stated'}\n\n"
                f"Candidates:\n{listing}"
            ),
            context={"candidates": candidates, "target": target},
            max_tokens=1536,
        )
        errors: list[str] = []
    except LLMError as exc:
        raw, errors = {"options": []}, [f"accommodation: {exc}"]

    chosen: list[LodgingOption] = []
    for pick in raw.get("options") or []:
        source = by_name.get(str(pick.get("name", "")).strip())
        if source is None:
            errors.append(f"accommodation: dropped {pick.get('name')!r}, not on the shortlist")
            continue
        chosen.append(
            LodgingOption(
                name=source["name"],
                area=source["area"],
                price_per_night_usd=source["price_per_night_usd"],
                rating=source["rating"],
                why=str(pick.get("why") or source["why"]),
            )
        )

    # If the model picked nothing usable, fall back to the cheapest that fit.
    if not chosen:
        ceiling = target * BUDGET_TOLERANCE
        affordable = [c for c in candidates if c["price_per_night_usd"] <= ceiling]
        chosen = [
            LodgingOption(**c)
            for c in sorted(affordable, key=lambda c: c["price_per_night_usd"])[:3]
        ]

    if params.budget and any(o.price_per_night_usd > target * BUDGET_TOLERANCE for o in chosen):
        errors.append(
            f"accommodation: a selected stay exceeds the {params.budget} target of ${target:.0f}"
        )

    return {
        "accommodation": AccommodationOutput(options=chosen, nightly_budget_usd=target),
        "sources": [f"{result.provider}:{result.source}"],
        "errors": errors,
        "trace": ["accommodation"],
    }
