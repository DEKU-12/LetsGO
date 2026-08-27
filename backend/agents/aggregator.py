"""Response Aggregator — merges every agent's output into one markdown plan.

The body of the plan is assembled deterministically from the structured state:
nothing appears in the final document that some agent did not put into state
first, which is what keeps the output auditable. The model is used only for a
short opening paragraph, and only from facts already in state.
"""

from __future__ import annotations

from typing import Any

from backend.llm import LLM, LLMError, register_mock
from backend.state import TravelState

SUMMARY_SYSTEM = """You write the opening paragraph of a travel plan.

Return JSON: {"summary": str}

Rules:
- Two or three sentences, addressed to the traveller.
- Use only facts given to you. Do not name places, prices or dates that are absent.
- No headings, no bullet points, no markdown."""


@register_mock("plan_summary")
def _mock_summary(context: dict[str, Any]) -> dict[str, Any]:
    """Template summary for the no-API-key path.

    Only mentions sections that were actually produced, so a weather question
    does not come back promising an itinerary and a hotel shortlist.
    """
    destination = context.get("destination", "your destination")
    days = context.get("duration_days")
    prefs = context.get("preferences") or []
    length = f"{days}-day " if days else ""
    interests = f", weighted toward {' and '.join(prefs[:2])}" if prefs else ""

    sentences = [f"Here is a {length}plan for {destination}{interests}."]
    if context.get("days_planned"):
        sentences.append(
            "The day-by-day schedule groups nearby stops together to keep travel time down."
        )
    if context.get("budget"):
        sentences.append(f"Everything below is pitched at a {context['budget']} budget.")
    if context.get("weather"):
        sentences.append("Check the weather note before locking in outdoor days.")

    return {"summary": " ".join(sentences)}


def _fmt_money(value: float | None) -> str:
    return f"${value:,.0f}" if value else "—"


def aggregator(state: TravelState, llm: LLM) -> dict[str, Any]:
    """Graph node: render the final markdown plan."""
    params = state["params"]
    assert params is not None

    sources = sorted(set(state.get("sources") or []))
    mocked = {s.split(":")[0] for s in sources if s.endswith(":mock")}

    research = state.get("research")
    itinerary = state.get("itinerary")
    accommodation = state.get("accommodation")
    transport = state.get("transport")

    facts = {
        "destination": params.destination,
        "duration_days": params.duration_days,
        "budget": params.budget,
        "travelers": params.travelers,
        "preferences": params.preferences,
        "weather": research.weather.summary if research and research.weather else None,
        "days_planned": len(itinerary.days) if itinerary else 0,
    }

    try:
        summary = str(
            llm.json(
                task="plan_summary",
                system=SUMMARY_SYSTEM,
                prompt=f"Facts available:\n{facts}",
                context=facts,
                max_tokens=512,
            ).get("summary", "")
        ).strip()
        errors: list[str] = []
    except LLMError as exc:
        summary = ""
        errors = [f"aggregator: {exc}"]

    title = f"# {params.destination} — {params.duration_days}-day plan" if params.duration_days \
        else f"# {params.destination} — travel plan"
    lines: list[str] = [title, ""]

    if summary:
        lines += [summary, ""]

    if research:
        lines.append("## The destination")
        lines.append("")
        if research.weather:
            w = research.weather
            temps = ""
            if w.avg_high_c is not None and w.avg_low_c is not None:
                low, high = round(w.avg_low_c), round(w.avg_high_c)
                temps = (
                    f" Around {high}°C."
                    if low == high
                    else f" Typical range {low}–{high}°C."
                )
            lines += [f"**Weather.** {w.summary}{temps} {w.advice}".strip(), ""]
        if research.attractions:
            lines.append("**Worth your time**")
            lines.append("")
            for a in research.attractions:
                detail = f" — {a.description}" if a.description else ""
                lines.append(f"- **{a.name}** _({a.category}, ~{a.est_hours:g}h)_{detail}")
            lines.append("")
        if research.practical_notes:
            lines.append("**Practical notes**")
            lines.append("")
            lines += [f"- {n}" for n in research.practical_notes]
            lines.append("")

    if itinerary and itinerary.days:
        lines += ["## Day by day", ""]
        for day in itinerary.days:
            heading = f"### Day {day.day}"
            if day.theme:
                heading += f" — {day.theme}"
            lines += [heading, ""]
            for block in day.blocks:
                detail = f" — {block.detail}" if block.detail else ""
                lines.append(f"- **{block.time}** {block.title}{detail}")
            lines.append("")

    if accommodation and accommodation.options:
        lines += ["## Where to stay", ""]
        if accommodation.nightly_budget_usd:
            lines += [
                f"Target nightly spend: {_fmt_money(accommodation.nightly_budget_usd)}.",
                "",
            ]
        if "lodging" in mocked:
            lines += [
                "_No live hotel inventory is connected — these properties and "
                "prices are illustrative, not bookable._",
                "",
            ]
        lines += ["| Option | Area | Per night | Rating | Why |", "| --- | --- | --- | --- | --- |"]
        for o in accommodation.options:
            rating = f"{o.rating:.1f}" if o.rating else "—"
            lines.append(
                f"| {o.name} | {o.area or '—'} | {_fmt_money(o.price_per_night_usd)} | {rating} | {o.why or '—'} |"
            )
        lines.append("")

    if transport and (transport.inbound or transport.local):
        lines += ["## Getting there and around", ""]
        if "transport" in mocked:
            lines += [
                "_No live fare data is connected — costs below are typical bands, "
                "not quotes._",
                "",
            ]
        if transport.inbound:
            lines += ["**Getting there**", ""]
            for leg in transport.inbound:
                cost = f" ({_fmt_money(leg.est_cost_usd)})" if leg.est_cost_usd else ""
                lines.append(f"- **{leg.mode}** — {leg.description}{cost}")
            lines.append("")
        if transport.local:
            lines += ["**Getting around**", ""]
            for leg in transport.local:
                cost = f" ({_fmt_money(leg.est_cost_usd)})" if leg.est_cost_usd else ""
                lines.append(f"- **{leg.mode}** — {leg.description}{cost}")
            lines.append("")

    footer = []
    if research and research.attractions:
        checked = [a for a in research.attractions if a.verified is not None]
        if checked:
            confirmed = sum(a.verified for a in checked)
            footer.append(
                f"{confirmed} of {len(checked)} recommended places confirmed "
                "against an independent map dataset"
            )
    if sources:
        footer.append(f"Data sources: {', '.join(sources)}")
    if footer:
        lines += ["---", "", f"_{'. '.join(footer)}._"]

    return {"final_plan": "\n".join(lines).strip(), "trace": ["aggregator"], "errors": errors}
