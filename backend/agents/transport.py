"""Transport agent — how the traveller gets there and gets around.

Options come from the transport adapter (mock by necessity — real fare data is
gated). The model narrows and annotates them for this specific trip; it does
not price flights, because a made-up fare stated confidently is worse than an
honest band.
"""

from __future__ import annotations

from typing import Any

from backend.adapters.transport import TransportAdapter
from backend.llm import LLM, LLMError, register_mock
from backend.state import TransportLeg, TransportOutput, TravelState

TRANSPORT_SYSTEM = """You advise on travel to and around a destination.

Return JSON:
  {"local": [{"mode": str, "description": str}], "note": str}

Rules:
- Keep 2 to 4 local transport modes from the candidates you are given, using the
  same mode names, and rewrite each description for this specific trip length
  and group size.
- Do not invent fares. Costs are attached separately.
- "note" is one short practical line — what to buy on arrival, or what to skip."""


def _change(state: TravelState) -> str:
    """The traveller's change request, when this agent is rerun to edit a plan."""
    request = state.get("edit_request")
    return f"\n\nThe traveller asked for this change: {request}" if request else ""


@register_mock("transport")
def _mock_transport(context: dict[str, Any]) -> dict[str, Any]:
    candidates = context.get("local") or []
    return {
        "local": [{"mode": c["mode"], "description": c["description"]} for c in candidates[:4]],
        "note": "Buy any multi-day transit pass at the airport on arrival.",
    }


def transport(state: TravelState, llm: LLM) -> dict[str, Any]:
    """Graph node: propose how to get there and get around."""
    params = state["params"]
    assert params is not None

    result = TransportAdapter().fetch(
        destination=params.destination, travelers=params.travelers
    )
    inbound: list[dict[str, Any]] = result.data["inbound"]
    local: list[dict[str, Any]] = result.data["local"]
    by_mode = {c["mode"]: c for c in local}

    listing = "\n".join(f"- {c['mode']}: {c['description']}" for c in local)

    try:
        raw = llm.json(
            task="transport",
            system=TRANSPORT_SYSTEM,
            prompt=(
                f"Destination: {params.destination}\n"
                f"Trip length: {params.duration_days or 'unspecified'} days\n"
                f"Travellers: {params.travelers}\n\n"
                f"Local transport candidates:\n{listing}"
                f"{_change(state)}"
            ),
            context={"local": local},
            max_tokens=1024,
        )
        errors: list[str] = []
    except LLMError as exc:
        raw, errors = {"local": []}, [f"transport: {exc}"]

    chosen: list[TransportLeg] = []
    for pick in raw.get("local") or []:
        source = by_mode.get(str(pick.get("mode", "")).strip())
        if source is None:
            errors.append(f"transport: dropped {pick.get('mode')!r}, not a known mode")
            continue
        chosen.append(
            TransportLeg(
                mode=source["mode"],
                description=str(pick.get("description") or source["description"]),
                est_cost_usd=source["est_cost_usd"],
            )
        )

    if not chosen:
        chosen = [TransportLeg(**c) for c in local]

    return {
        "transport": TransportOutput(
            inbound=[TransportLeg(**leg) for leg in inbound],
            local=chosen,
        ),
        "sources": [f"{result.provider}:{result.source}"],
        "errors": errors,
        "trace": ["transport"],
    }
