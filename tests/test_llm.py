"""The model wrapper: provider selection, JSON parsing, mock registry."""

from __future__ import annotations

import pytest

from backend.llm import LLM, LLMError, _first_json_object, register_mock


def test_mock_provider_needs_no_key() -> None:
    llm = LLM(provider="mock")
    assert llm.is_mock
    assert llm.model == "mock-deterministic"


def test_unknown_provider_is_rejected() -> None:
    with pytest.raises(LLMError):
        LLM(provider="not-a-provider")


def test_unregistered_mock_task_fails_loudly() -> None:
    with pytest.raises(LLMError, match="no mock registered"):
        LLM(provider="mock").json(task="nope", system="", prompt="")


def test_registered_mock_task_is_returned() -> None:
    @register_mock("test_task")
    def _mock(context):
        return {"echo": context.get("value")}

    assert LLM(provider="mock").json(
        task="test_task", system="", prompt="", context={"value": 42}
    ) == {"echo": 42}


@pytest.mark.parametrize(
    "raw",
    [
        '{"a": 1}',
        '```json\n{"a": 1}\n```',
        'Sure, here you go:\n```\n{"a": 1}\n```\nHope that helps!',
    ],
)
def test_json_is_recovered_from_chatty_or_fenced_replies(raw: str) -> None:
    import json

    assert json.loads(_first_json_object(raw)) == {"a": 1}


def test_provider_failure_becomes_an_llm_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A dead key must surface as LLMError so nodes degrade instead of crashing."""
    llm = LLM(provider="mock")
    llm.is_mock = False
    class DeadKey:
        tokens = [0, 0]  # every backend counts tokens

        def __call__(self, *_):
            raise RuntimeError("401 Invalid API Key")

    llm._backend = DeadKey()

    with pytest.raises(LLMError, match="401 Invalid API Key"):
        llm.text(task="anything", system="", prompt="")
