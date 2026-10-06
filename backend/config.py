"""Runtime configuration, read once from the environment.

Nothing here is required. Every key is optional and every consumer has a mock
fallback, so a fresh clone runs end to end with an empty environment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# override=True: the project's .env wins over the shell. A key exported in a
# shell profile for some other project would otherwise silently shadow the one
# in .env, and every run fails with an auth error that looks like a model
# problem. Without a .env file (CI, deploys) the environment is used as is.
load_dotenv(PROJECT_ROOT / ".env", override=True)


def _env(name: str) -> str | None:
    value = os.getenv(name, "").strip()
    return value or None


@dataclass(frozen=True)
class Settings:
    """Immutable snapshot of the environment."""

    anthropic_api_key: str | None = field(default_factory=lambda: _env("ANTHROPIC_API_KEY"))
    # Identity-linked keys (sk-ant-api03-...) must name the workspace they
    # act in; workspace-scoped keys do not need this.
    anthropic_workspace_id: str | None = field(
        default_factory=lambda: _env("ANTHROPIC_WORKSPACE_ID")
    )
    groq_api_key: str | None = field(default_factory=lambda: _env("GROQ_API_KEY"))
    llm_provider: str = field(default_factory=lambda: _env("LLM_PROVIDER") or "auto")
    llm_model: str | None = field(default_factory=lambda: _env("LLM_MODEL"))

    openweather_api_key: str | None = field(default_factory=lambda: _env("OPENWEATHER_API_KEY"))
    geoapify_api_key: str | None = field(default_factory=lambda: _env("GEOAPIFY_API_KEY"))
    opentripmap_api_key: str | None = field(default_factory=lambda: _env("OPENTRIPMAP_API_KEY"))
    google_places_api_key: str | None = field(default_factory=lambda: _env("GOOGLE_PLACES_API_KEY"))

    database_url: str = field(
        default_factory=lambda: _env("DATABASE_URL") or "sqlite:///./letsgo.db"
    )

    langsmith_api_key: str | None = field(default_factory=lambda: _env("LANGSMITH_API_KEY"))
    langsmith_tracing: bool = field(
        default_factory=lambda: (_env("LANGSMITH_TRACING") or "").lower() == "true"
    )
    langsmith_project: str = field(default_factory=lambda: _env("LANGSMITH_PROJECT") or "letsgo")

    def resolved_provider(self) -> str:
        """Pick a concrete LLM backend.

        `auto` prefers the provider the spec asks for (Anthropic), falls back to
        Groq if that is the only key present, and finally to the deterministic
        mock so the system always runs.
        """
        if self.llm_provider != "auto":
            return self.llm_provider
        if self.anthropic_api_key:
            return "anthropic"
        if self.groq_api_key:
            return "groq"
        return "mock"


settings = Settings()
