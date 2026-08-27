"""Storage for finished trips and the runs that produced them.

SQLite by default so a fresh clone needs no database server; point
``DATABASE_URL`` at Postgres for anything real. Two tables:

* ``trips`` — the request, the parsed parameters, the finished plan.
* ``agent_runs`` — which agents ran, in what order, and what they recorded.

The second table exists because the trajectory is the interesting part of a
multi-agent system, and a plan with no record of how it was produced cannot be
audited after the fact.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    select,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship

from backend.config import settings
from backend.state import TravelState


class Base(DeclarativeBase):
    pass


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Trip(Base):
    __tablename__ = "trips"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)
    request: Mapped[str] = mapped_column(Text)
    destination: Mapped[str | None] = mapped_column(String(200), nullable=True)
    params: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    plan: Mapped[str | None] = mapped_column(Text, nullable=True)
    clarification: Mapped[str | None] = mapped_column(Text, nullable=True)
    provider: Mapped[str] = mapped_column(String(40))
    model: Mapped[str] = mapped_column(String(120))
    duration_s: Mapped[float] = mapped_column(Float, default=0.0)
    sources: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)

    runs: Mapped[list[AgentRun]] = relationship(
        back_populates="trip", cascade="all, delete-orphan", order_by="AgentRun.position"
    )

    def summary(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "created_at": self.created_at.isoformat(),
            "request": self.request,
            "destination": self.destination,
            "provider": self.provider,
            "model": self.model,
            "duration_s": round(self.duration_s, 1),
            "agents": [r.agent for r in self.runs],
            "has_plan": bool(self.plan),
        }

    def detail(self) -> dict[str, Any]:
        return {
            **self.summary(),
            "params": self.params,
            "plan": self.plan,
            "clarification": self.clarification,
            "sources": self.sources or [],
            "runs": [
                {"agent": r.agent, "position": r.position, "errors": r.errors or []}
                for r in self.runs
            ],
        }


class AgentRun(Base):
    __tablename__ = "agent_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    trip_id: Mapped[int] = mapped_column(ForeignKey("trips.id", ondelete="CASCADE"))
    agent: Mapped[str] = mapped_column(String(60))
    position: Mapped[int] = mapped_column(Integer)
    errors: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)

    trip: Mapped[Trip] = relationship(back_populates="runs")


_engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False} if settings.database_url.startswith("sqlite") else {},
)


def init_db() -> None:
    Base.metadata.create_all(_engine)


@contextmanager
def session_scope() -> Iterator[Session]:
    session = Session(_engine)
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def save_trip(
    state: TravelState, *, provider: str, model: str, duration_s: float
) -> int:
    """Persist one finished run. Returns the new trip id."""
    params = state.get("params")

    with session_scope() as session:
        trip = Trip(
            request=state.get("request", ""),
            destination=params.destination if params else None,
            params=json.loads(params.model_dump_json()) if params else None,
            plan=state.get("final_plan"),
            clarification=state.get("clarification"),
            provider=provider,
            model=model,
            duration_s=duration_s,
            sources=sorted(set(state.get("sources") or [])),
        )

        errors = state.get("errors") or []
        for position, step in enumerate(state.get("trace") or []):
            trip.runs.append(
                AgentRun(
                    agent=step,
                    position=position,
                    errors=[e for e in errors if e.startswith(f"{step}:")],
                )
            )

        session.add(trip)
        session.flush()
        return trip.id


def list_trips(limit: int = 50) -> list[dict[str, Any]]:
    with session_scope() as session:
        rows = session.scalars(
            select(Trip).order_by(Trip.created_at.desc()).limit(limit)
        ).all()
        return [row.summary() for row in rows]


def get_trip(trip_id: int) -> dict[str, Any] | None:
    with session_scope() as session:
        trip = session.get(Trip, trip_id)
        return trip.detail() if trip else None
