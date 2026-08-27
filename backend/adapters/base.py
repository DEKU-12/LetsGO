"""Common shape for every external data source.

Two rules hold for all adapters:

1. One interface — callers use :meth:`Adapter.fetch` and never touch HTTP.
2. Always a mock — with no API key (or when a live call fails) the adapter
   returns realistic canned data tagged ``source="mock"``, so the system runs
   end to end with zero keys and the demo never dies on a network error.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class AdapterResult:
    """Data plus provenance, so downstream output can say where it came from."""

    data: Any
    source: Literal["live", "mock"]
    provider: str

    @property
    def is_mock(self) -> bool:
        return self.source == "mock"


class Adapter(ABC):
    """Base class: subclasses implement `_fetch_live` and `_fetch_mock`."""

    name: str = "adapter"

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key

    @property
    def live(self) -> bool:
        return bool(self.api_key)

    def fetch(self, **kwargs: Any) -> AdapterResult:
        if self.live:
            try:
                return AdapterResult(self._fetch_live(**kwargs), "live", self.name)
            except Exception as exc:  # noqa: BLE001 - degrade, never crash the run
                log.warning("%s: live call failed (%s); falling back to mock", self.name, exc)
        return AdapterResult(self._fetch_mock(**kwargs), "mock", self.name)

    @abstractmethod
    def _fetch_live(self, **kwargs: Any) -> Any: ...

    @abstractmethod
    def _fetch_mock(self, **kwargs: Any) -> Any: ...
