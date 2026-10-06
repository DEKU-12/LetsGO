"""A failed itinerary must show up in the plan and in the errors, not hide."""

from __future__ import annotations

from typing import Any

from backend.graph import plan_trip
from backend.llm import LLM, LLMError


def test_failed_itinerary_is_flagged(monkeypatch) -> None:
    original = LLM.json

    def failing(self: LLM, **kwargs: Any) -> Any:
        if kwargs["task"] == "itinerary":
            raise LLMError(f"{self.provider}: RateLimitError: tokens per day")
        return original(self, **kwargs)

    monkeypatch.setattr(LLM, "json", failing)
    state = plan_trip("5 days in Japan, love food and history", LLM(provider="mock"))

    assert "could not be generated" in state["final_plan"]
    assert any(e.startswith("validate_plan:") and "itinerary" in e for e in state["errors"])
