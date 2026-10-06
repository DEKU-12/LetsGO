"""Thin, swappable wrapper around the chat model.

The spec calls for Claude (`claude-sonnet` class). Two extra backends exist for
practical reasons and both are selected automatically:

* ``groq``  — a free-tier fallback so the project is cheap to iterate on.
* ``mock``  — deterministic canned responses so the whole system runs, and the
  test suite passes, with zero API keys.

Every call goes through :meth:`LLM.json` or :meth:`LLM.text`, so swapping the
provider is a one-line change and eval runs are cheap to configure.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable

from backend.config import settings

DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-5",
    # Groq retired the Llama 3.3 endpoints; gpt-oss-120b is the current
    # free-tier model that follows JSON instructions reliably.
    "groq": "openai/gpt-oss-120b",
    "mock": "mock-deterministic",
}

_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.MULTILINE)


class LLMError(RuntimeError):
    """Raised when the model could not produce a usable response."""


def _strip_fences(raw: str) -> str:
    return _FENCE.sub("", raw).strip()


def _first_json_object(raw: str) -> str:
    """Pull the outermost JSON value out of a chatty response."""
    text = _strip_fences(raw)
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        end = text.rfind(closer)
        if start != -1 and end > start:
            return text[start : end + 1]
    return text


# --------------------------------------------------------------------------
# Backends
# --------------------------------------------------------------------------


#: Retries on rate limits and transient errors. The SDKs wait between tries,
#: honouring the provider's retry-after hint, but stop after 2 by default.
#: Groq's free tier resets its per-minute token budget once a minute, and 8
#: tries with the SDK's backoff covers that window.
MAX_RETRIES = 8


@dataclass
class _Anthropic:
    model: str

    def __post_init__(self) -> None:
        import anthropic  # imported lazily so the mock path needs no SDK

        headers = {}
        if settings.anthropic_workspace_id:
            headers["anthropic-workspace-id"] = settings.anthropic_workspace_id

        self._client = anthropic.Anthropic(
            api_key=settings.anthropic_api_key,
            # A stray ANTHROPIC_BASE_URL in the shell would otherwise send
            # these requests somewhere unexpected.
            base_url="https://api.anthropic.com",
            default_headers=headers or None,
            max_retries=MAX_RETRIES,
        )

    def __call__(self, system: str, prompt: str, max_tokens: int) -> str:
        response = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in response.content if b.type == "text")


@dataclass
class _Groq:
    model: str

    def __post_init__(self) -> None:
        from groq import Groq

        self._client = Groq(api_key=settings.groq_api_key, max_retries=MAX_RETRIES)

    def __call__(self, system: str, prompt: str, max_tokens: int) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=0,  # reproducible eval runs
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content or ""


@dataclass
class _Mock:
    model: str

    def __call__(self, system: str, prompt: str, max_tokens: int) -> str:  # pragma: no cover
        raise LLMError(
            "the mock backend answers by task name, not by prompt — "
            "call LLM.json()/LLM.text() with a registered task"
        )


# --------------------------------------------------------------------------
# Mock registry
# --------------------------------------------------------------------------

MockFn = Callable[[dict[str, Any]], Any]
_MOCKS: dict[str, MockFn] = {}


def register_mock(task: str) -> Callable[[MockFn], MockFn]:
    """Register the canned response for `task` used when no API key is set."""

    def decorator(fn: MockFn) -> MockFn:
        _MOCKS[task] = fn
        return fn

    return decorator


# --------------------------------------------------------------------------
# Public wrapper
# --------------------------------------------------------------------------


class LLM:
    """The single entry point every agent uses to talk to a model."""

    def __init__(self, provider: str | None = None, model: str | None = None) -> None:
        self.provider = provider or settings.resolved_provider()
        if self.provider not in DEFAULT_MODELS:
            raise LLMError(f"unknown LLM provider {self.provider!r}")
        self.model = model or settings.llm_model or DEFAULT_MODELS[self.provider]
        self.is_mock = self.provider == "mock"
        backend_cls = {"anthropic": _Anthropic, "groq": _Groq, "mock": _Mock}[self.provider]
        try:
            self._backend = backend_cls(self.model)
        except Exception as exc:  # noqa: BLE001 - a bad client config is an LLM error
            raise LLMError(f"could not initialise {self.provider} backend: {exc}") from exc

    def __repr__(self) -> str:
        return f"LLM(provider={self.provider!r}, model={self.model!r})"

    # -- raw text ----------------------------------------------------------

    def text(
        self,
        *,
        task: str,
        system: str,
        prompt: str,
        context: dict[str, Any] | None = None,
        max_tokens: int = 4096,
    ) -> str:
        if self.is_mock:
            return str(self._mock(task, context or {}))
        return self._call(system, prompt, max_tokens)

    # -- JSON --------------------------------------------------------------

    def json(
        self,
        *,
        task: str,
        system: str,
        prompt: str,
        context: dict[str, Any] | None = None,
        max_tokens: int = 4096,
    ) -> Any:
        """Call the model and parse a JSON body, retrying once on a parse failure."""
        if self.is_mock:
            return self._mock(task, context or {})

        system = f"{system}\n\nRespond with JSON only. No prose, no code fences."
        attempt_prompt = prompt
        last_error: Exception | None = None

        for _ in range(2):
            raw = self._call(system, attempt_prompt, max_tokens)
            try:
                return json.loads(_first_json_object(raw))
            except (json.JSONDecodeError, ValueError) as exc:
                last_error = exc
                attempt_prompt = (
                    f"{prompt}\n\nYour previous reply was not valid JSON "
                    f"({exc}). Return the JSON value only."
                )

        raise LLMError(f"{task}: model did not return valid JSON ({last_error})")

    # -- transport ---------------------------------------------------------

    def _call(self, system: str, prompt: str, max_tokens: int) -> str:
        """Single choke point for provider errors.

        Provider failures — a dead key, a rate limit, a network blip — surface as
        `LLMError`. Every node catches that and records it in `state["errors"]`,
        so a bad credential degrades the run and says why instead of crashing it.
        """
        try:
            return self._backend(system, prompt, max_tokens)
        except Exception as exc:  # noqa: BLE001 - provider SDKs raise their own types
            raise LLMError(f"{self.provider}: {type(exc).__name__}: {exc}") from exc

    # -- mocks -------------------------------------------------------------

    def _mock(self, task: str, context: dict[str, Any]) -> Any:
        try:
            return _MOCKS[task](context)
        except KeyError:
            raise LLMError(
                f"no mock registered for task {task!r}; add one with @register_mock"
            ) from None
