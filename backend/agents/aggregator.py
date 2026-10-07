"""Response Aggregator — merges every agent's output into one markdown plan.

The body of the plan is assembled deterministically from the structured state:
nothing appears in the final document that some agent did not put into state
first, which is what keeps the output auditable. The model is used only for a
short opening paragraph, and only from facts already in state.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import quote

from backend.llm import LLM, LLMError, register_mock
from backend.state import ITINERARY, Photo, TravelState

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
    does not come back promising an itinerary and places to stay.
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


def _url(url: str) -> str:
    """Commons file names contain parentheses, which end a markdown link."""
    return quote(url, safe=":/%")


def _photo_md(name: str, photo: Photo) -> tuple[str, str]:
    """(image, credit) markdown. The credit links to the Commons file page,
    which carries the author and licence the photo is used under."""
    alt = name.replace("[", "").replace("]", "")
    if photo.generic:
        alt = f"{photo.caption} (general photo)"
        shows = f" — general photo of {photo.caption}, not this place"
    else:
        shows = f" — shows {photo.caption}" if photo.caption else ""
    credit = f"_[Photo: {photo.credit or 'Wikimedia Commons'}]({_url(photo.page)}){shows}_"
    return f"![{alt}]({_url(photo.url)})", credit


def aggregator(state: TravelState, llm: LLM) -> dict[str, Any]:
    """Graph node: render the final markdown plan."""
    params = state["params"]
    assert params is not None

    sources = sorted(set(state.get("sources") or []))

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
                effort="low",
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

    if research and research.photo:
        image, credit = _photo_md(params.destination, research.photo)
        lines += [image, "", credit, ""]

    if summary:
        lines += [summary, ""]

    if research:
        lines.append("## The destination")
        lines.append("")
        # One line per city the research agent checked, or just the one.
        outlooks = research.city_weather if len(research.city_weather) > 1 else (
            {"": research.weather} if research.weather else {}
        )
        for city, w in outlooks.items():
            temps = ""
            if w.avg_high_c is not None and w.avg_low_c is not None:
                low, high = round(w.avg_low_c), round(w.avg_high_c)
                temps = (
                    f" Around {high}°C."
                    if low == high
                    else f" Typical range {low}–{high}°C."
                )
            label = f"Weather in {city}." if city else "Weather."
            lines += [f"**{label}** {w.summary}{temps} {w.advice}".strip(), ""]
        if research.attractions:
            lines.append("**Worth your time**")
            lines.append("")
            for a in research.attractions:
                detail = f" — {a.description}" if a.description else ""
                hours = f", open {a.opening_hours}" if a.opening_hours else ""
                image, credit = _photo_md(a.name, a.photo) if a.photo else ("", "")
                lines.append(
                    f"- {image + ' ' if image else ''}**{a.name}** "
                    f"_({a.category}, ~{a.est_hours:g}h{hours})_{detail}"
                    f"{' ' + credit if credit else ''}"
                )
            lines.append("")
        if research.practical_notes:
            lines.append("**Practical notes**")
            lines.append("")
            lines += [f"- {n}" for n in research.practical_notes]
            lines += ["", "_General guidance from the model, not checked against official "
                      "sources. Confirm visa and entry rules with the official government "
                      "site before you travel._", ""]

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
    elif ITINERARY in (state.get("route_plan") or []):
        # Asked for but nothing came back: label the gap in the plan itself,
        # so the reader does not miss it.
        lines += [
            "## Day by day",
            "",
            "_The day-by-day schedule could not be generated this time — "
            "try again in a minute._",
            "",
        ]

    if accommodation and accommodation.areas:
        lines += ["## Where to stay", ""]
        for area in accommodation.areas:
            level = f" _({area.price_level})_" if area.price_level else ""
            why = f" — {area.why}" if area.why else ""
            lines.append(f"- **{area.name}**{level}{why}")
        lines.append("")
        lines += [f"- {t}" for t in accommodation.tips]
        lines += ["", "_Areas, not listings: pick a place on your usual booking site._", ""]

    if transport and (transport.arrival or transport.local):
        lines += ["## Getting there and around", ""]
        if transport.arrival:
            lines += ["**Arriving**", ""]
            lines += [f"- {a}" for a in transport.arrival]
            lines.append("")
        if transport.local:
            lines += ["**Getting around**", ""]
            lines += [f"- **{t.mode}** — {t.description}" for t in transport.local]
            lines.append("")
        if transport.tips:
            lines += [f"- {t}" for t in transport.tips]
            lines.append("")

    footer = []
    named = [*(research.attractions if research else []),
             *(accommodation.areas if accommodation else [])]
    checked = [p for p in named if p.verified is not None]
    if checked:
        confirmed = sum(p.verified for p in checked)
        footer.append(
            f"{confirmed} of {len(checked)} recommended places confirmed "
            "against an independent map dataset"
        )
    if sources:
        footer.append(f"Data sources: {', '.join(sources)}")
    if footer:
        lines += ["---", "", f"_{'. '.join(footer)}._"]

    return {"final_plan": "\n".join(lines).strip(), "trace": ["aggregator"], "errors": errors}
