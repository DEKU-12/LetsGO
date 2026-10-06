"""Tool calling: the wrapper's loop, and the research agent's use of it."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from backend.agents.destination_research import destination_research
from backend.llm import LLM, Tool
from backend.state import TripParams


def _weather_tool(log: list[str]) -> Tool:
    return Tool(
        "get_weather", "weather",
        {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
        lambda city: log.append(city) or {"summary": f"sunny in {city}"},
    )


class _ScriptedGroq:
    """Stands in for the Groq SDK: replays tool calls, then a final answer."""

    def __init__(self, turns: list[list[tuple[str, str]] | str]) -> None:
        self.turns, self.seen = list(turns), []

    def create(self, **kwargs: Any) -> Any:
        self.seen.append(kwargs)
        turn = self.turns.pop(0)
        if isinstance(turn, str):
            message = SimpleNamespace(content=turn, tool_calls=None)
        else:
            message = SimpleNamespace(content="", tool_calls=[
                SimpleNamespace(id=f"c{i}", function=SimpleNamespace(name=n, arguments=a))
                for i, (n, a) in enumerate(turn)
            ])
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _groq_llm(monkeypatch, turns) -> tuple[LLM, _ScriptedGroq]:
    # A dummy key: the SDK client is replaced below, so nothing reaches Groq.
    monkeypatch.setattr(
        "backend.llm.settings",
        SimpleNamespace(groq_api_key="test", llm_model=None, resolved_provider=lambda: "groq"),
    )
    llm = LLM(provider="groq")
    fake = _ScriptedGroq(turns)
    llm._backend._client = SimpleNamespace(chat=SimpleNamespace(completions=fake))
    return llm, fake


def test_the_model_chooses_calls_and_sees_results(monkeypatch) -> None:
    called: list[str] = []
    llm, fake = _groq_llm(monkeypatch, [
        [("get_weather", '{"city": "Tokyo"}'), ("get_weather", '{"city": "Kyoto"}')],
        '{"done": true}',
    ])

    answer, calls = llm.run_tools(task="t", system="s", prompt="p", tools=[_weather_tool(called)])

    assert answer == {"done": True}
    assert called == ["Tokyo", "Kyoto"]
    assert [c["args"] for c in calls] == [{"city": "Tokyo"}, {"city": "Kyoto"}]
    # The results went back to the model before it answered.
    tool_messages = [m for m in fake.seen[1]["messages"] if m["role"] == "tool"]
    assert json.loads(tool_messages[0]["content"]) == {"summary": "sunny in Tokyo"}


def test_bad_calls_are_logged_and_reported_back_not_crashes(monkeypatch) -> None:
    llm, fake = _groq_llm(monkeypatch, [
        [("get_flights", "{}"), ("get_weather", '{"town": "Rome"}')],
        '{"done": true}',
    ])

    _, calls = llm.run_tools(task="t", system="s", prompt="p", tools=[_weather_tool([])])

    assert [c["ok"] for c in calls] == [False, False]
    assert "unknown tool" in calls[0]["error"]
    assert "town" in calls[1]["error"]  # wrong argument name
    errors = [json.loads(m["content"]) for m in fake.seen[1]["messages"] if m["role"] == "tool"]
    assert all("error" in e for e in errors)


def test_the_last_round_forbids_more_tool_calls(monkeypatch) -> None:
    llm, fake = _groq_llm(monkeypatch, [
        [("get_weather", '{"city": "A"}')],
        [("get_weather", '{"city": "B"}')],
        '{"done": true}',
    ])

    llm.run_tools(task="t", system="s", prompt="p", tools=[_weather_tool([])], max_rounds=2)

    assert [k.get("tool_choice") for k in fake.seen] == ["auto", "auto", None]
    assert "tools" not in fake.seen[-1]
    assert "used all your lookups" in fake.seen[-1]["messages"][-1]["content"]


# -- the research agent -------------------------------------------------------


@pytest.fixture
def state() -> dict[str, Any]:
    return {"request": "4 days in Lisbon", "meta": {"routing": {"plan": []}},
            "params": TripParams(destination="Lisbon", duration_days=4)}


def test_research_logs_its_tool_calls_and_keeps_existing_meta(state) -> None:
    update = destination_research(state, LLM(provider="mock"))

    calls = update["meta"]["tool_calls"]["destination_research"]
    assert [(c["name"], c["args"]) for c in calls] == [("get_weather", {"city": "Lisbon"})]
    assert update["meta"]["routing"] == {"plan": []}
    assert "Lisbon" in update["research"].city_weather


def test_weather_is_fetched_even_if_the_model_never_asks(state, monkeypatch) -> None:
    monkeypatch.setattr(
        LLM, "run_tools", lambda self, **kw: ({"attractions": [], "practical_notes": []}, [])
    )
    update = destination_research(state, LLM(provider="mock"))

    assert update["research"].weather is not None
    calls = update["meta"]["tool_calls"]["destination_research"]
    assert calls == [{"name": "get_weather", "args": {"city": "Lisbon"}, "ok": True, "fallback": True}]
