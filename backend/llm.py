"""Thin, swappable wrapper around the chat model.

The spec calls for Claude (`claude-sonnet` class). Two extra backends exist for
practical reasons and both are selected automatically:

* ``groq``  — a free-tier fallback so the project is cheap to iterate on.
* ``mock``  — deterministic canned responses so the whole system runs, and the
  test suite passes, with zero API keys.

Every call goes through :meth:`LLM.json`, :meth:`LLM.text` or
:meth:`LLM.run_tools`, so swapping the provider is a one-line change and eval
runs are cheap to configure.

``run_tools`` is the one place the model decides what to call: it is given
:class:`Tool` definitions, chooses which to call and with what arguments, sees
the results, and finally answers in JSON. Every call it makes is logged, so the
choices can be evaluated (see ``eval/tools.py``).
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


@dataclass(frozen=True)
class Tool:
    """A function the model may choose to call.

    ``parameters`` is a JSON Schema object describing the arguments; ``run`` is
    called with them as keyword arguments and returns something JSON-encodable.
    """

    name: str
    description: str
    parameters: dict[str, Any]
    run: Callable[..., Any]


#: Rounds of tool calls before the model must answer. One round can hold
#: several calls, so this bounds cost, not what the model can look up.
MAX_TOOL_ROUNDS = 6

#: Sent when the tool rounds run out.
FINAL_ROUND = "You have used all your lookups. Answer now, using what you have found."

#: Executes one tool call: (name, raw arguments) -> JSON string result.
Execute = Callable[[str, Any], str]


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

    def run_tools(
        self, system: str, prompt: str, tools: list[Tool], execute: Execute,
        max_rounds: int, max_tokens: int,
    ) -> str:
        specs = [
            {"name": t.name, "description": t.description, "input_schema": t.parameters}
            for t in tools
        ]
        messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
        for round_ in range(max_rounds + 1):
            response = self._client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                messages=messages,
                tools=specs,
                # Last round: answer with what you have.
                tool_choice={"type": "none" if round_ == max_rounds else "auto"},
            )
            calls = [b for b in response.content if b.type == "tool_use"]
            if not calls:
                return "".join(b.text for b in response.content if b.type == "text")
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": c.id, "content": execute(c.name, c.input)}
                for c in calls
            ]})
        return ""


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

    def run_tools(
        self, system: str, prompt: str, tools: list[Tool], execute: Execute,
        max_rounds: int, max_tokens: int,
    ) -> str:
        specs = [
            {"type": "function", "function": {
                "name": t.name, "description": t.description, "parameters": t.parameters,
            }}
            for t in tools
        ]
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        for round_ in range(max_rounds + 1):
            last = round_ == max_rounds
            if last:
                # Groq rejects the whole request if the model tries a tool call
                # under tool_choice="none", so take the tools away instead and
                # say so plainly.
                messages.append({"role": "user", "content": FINAL_ROUND})
            response = self._client.chat.completions.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=0,
                messages=messages,
                **({} if last else {"tools": specs, "tool_choice": "auto"}),
            )
            message = response.choices[0].message
            if not message.tool_calls:
                return message.content or ""
            messages.append({
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {"id": c.id, "type": "function",
                     "function": {"name": c.function.name, "arguments": c.function.arguments}}
                    for c in message.tool_calls
                ],
            })
            for c in message.tool_calls:
                messages.append({
                    "role": "tool",
                    "tool_call_id": c.id,
                    "content": execute(c.function.name, c.function.arguments),
                })
        return ""


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


#: For tool-using tasks: which tool calls the mock "decides" to make. Each
#: returns a list of {"name": ..., "args": {...}}.
_MOCK_TOOL_CALLS: dict[str, MockFn] = {}


def register_mock_tool_calls(task: str) -> Callable[[MockFn], MockFn]:
    """Register the tool calls the mock backend makes for `task`."""

    def decorator(fn: MockFn) -> MockFn:
        _MOCK_TOOL_CALLS[task] = fn
        return fn

    return decorator


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

    # -- tools -------------------------------------------------------------

    def run_tools(
        self,
        *,
        task: str,
        system: str,
        prompt: str,
        tools: list[Tool],
        context: dict[str, Any] | None = None,
        max_rounds: int = MAX_TOOL_ROUNDS,
        max_tokens: int = 4096,
    ) -> tuple[Any, list[dict[str, Any]]]:
        """Let the model call tools, then return its JSON answer and the call log.

        The log has one entry per call the model made, in order:
        ``{"name", "args", "ok", "result" | "error"}``. A call to an unknown
        tool, or with arguments the tool rejects, is logged with ``ok: False``
        and the error goes back to the model so it can correct itself.
        """
        calls: list[dict[str, Any]] = []
        by_name = {t.name: t for t in tools}

        def execute(name: str, raw_args: Any) -> str:
            entry: dict[str, Any] = {"name": name, "args": raw_args, "ok": False}
            calls.append(entry)
            try:
                args = json.loads(raw_args or "{}") if isinstance(raw_args, str) else dict(raw_args or {})
                entry["args"] = args
                if name not in by_name:
                    raise ValueError(f"unknown tool {name!r}")
                result = by_name[name].run(**args)
            except Exception as exc:  # noqa: BLE001 - the model sees the error and can retry
                entry["error"] = f"{type(exc).__name__}: {exc}"
                return json.dumps({"error": entry["error"]})
            entry["ok"], entry["result"] = True, result
            return json.dumps(result, default=str)

        if self.is_mock:
            for call in _MOCK_TOOL_CALLS.get(task, lambda _: [])(context or {}):
                execute(call["name"], call["args"])
            return self._mock(task, context or {}), calls

        system = f"{system}\n\nWhen you have what you need, respond with JSON only. No prose, no code fences."
        try:
            raw = self._backend.run_tools(system, prompt, tools, execute, max_rounds, max_tokens)
        except Exception as exc:  # noqa: BLE001 - provider SDKs raise their own types
            raise LLMError(f"{self.provider}: {type(exc).__name__}: {exc}") from exc

        try:
            return json.loads(_first_json_object(raw)), calls
        except (json.JSONDecodeError, ValueError):
            # Rare: tools went fine but the final answer was not JSON. Ask once
            # more, plainly, with the tool results inlined — no new tool calls.
            results = json.dumps(
                [{k: c.get(k) for k in ("name", "args", "result", "error")} for c in calls],
                default=str,
            )
            return self.json(
                task=task, system=system,
                prompt=f"{prompt}\n\nTool results you already have:\n{results}",
                max_tokens=max_tokens,
            ), calls

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
