"""Storage for finished trips and the runs that produced them.

SQLite by default so a fresh clone needs no database server; point
``DATABASE_URL`` at Postgres for anything real. Two tables:

* ``trips`` — the request, the parsed parameters, the finished plan, and the
  full shared state the plan was built from, so a trip can be reopened and
  changed later rather than only reread.
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
    inspect,
    select,
    text,
)
from pydantic import TypeAdapter
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship

from backend.config import settings
from backend.state import TravelState

#: Converts the shared state, with its pydantic models, to plain JSON and back.
_STATE = TypeAdapter(TravelState)


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
    state: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    #: Set on a version made by editing another trip: the trip it was edited
    #: from, and the message that asked for the change. Following parent_id
    #: back is the undo history.
    #: The anonymous browser id that made this trip. Only that browser can list,
    #: read or edit it: trip ids count up, so without an owner anyone could
    #: read every visitor's trips by guessing numbers.
    user_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    parent_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    edit_message: Mapped[str | None] = mapped_column(Text, nullable=True)

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
            "parent_id": self.parent_id,
            "edit_message": self.edit_message,
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


class Profile(Base):
    """Preferences a traveller asked us to remember, keyed by an anonymous id.

    The id is a random string the browser makes up; there are no accounts. A
    preference is only stored after the traveller says yes to it.
    """

    __tablename__ = "profiles"

    user_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    preferences: Mapped[list[str]] = mapped_column(JSON, default=list)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_now, onupdate=_now)


_engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False} if settings.database_url.startswith("sqlite") else {},
)


def init_db() -> None:
    Base.metadata.create_all(_engine)
    # create_all never alters an existing table, so a database made before
    # these columns existed needs them added by hand.
    added = {"state": "JSON", "parent_id": "INTEGER", "edit_message": "TEXT",
             "user_id": "VARCHAR(64)"}
    columns = {c["name"] for c in inspect(_engine).get_columns("trips")}
    with _engine.begin() as conn:
        for name, kind in added.items():
            if name not in columns:
                conn.execute(text(f"ALTER TABLE trips ADD COLUMN {name} {kind}"))


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
    state: TravelState,
    *,
    provider: str,
    model: str,
    duration_s: float,
    parent_id: int | None = None,
    edit_message: str | None = None,
    user_id: str | None = None,
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
            state=_STATE.dump_python(state, mode="json"),
            parent_id=parent_id,
            edit_message=edit_message,
            user_id=user_id,
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


def list_trips(user_id: str, limit: int = 50) -> list[dict[str, Any]]:
    """One browser's trips, newest first."""
    with session_scope() as session:
        rows = session.scalars(
            select(Trip).where(Trip.user_id == user_id)
            .order_by(Trip.created_at.desc()).limit(limit)
        ).all()
        return [row.summary() for row in rows]


def get_trip(trip_id: int, user_id: str) -> dict[str, Any] | None:
    """A trip, if `user_id` owns it. Not found and not yours look the same."""
    with session_scope() as session:
        trip = session.get(Trip, trip_id)
        return trip.detail() if trip and trip.user_id == user_id else None


def owns(trip_id: int, user_id: str | None) -> bool:
    with session_scope() as session:
        trip = session.get(Trip, trip_id)
        return bool(user_id) and trip is not None and trip.user_id == user_id


def load_state(trip_id: int) -> TravelState | None:
    """The full shared state a saved trip was built from, models restored."""
    with session_scope() as session:
        trip = session.get(Trip, trip_id)
        if trip is None or trip.state is None:
            return None
        return _STATE.validate_python(trip.state)


def get_profile(user_id: str) -> list[str]:
    with session_scope() as session:
        profile = session.get(Profile, user_id)
        return list(profile.preferences) if profile else []


def set_profile(user_id: str, preferences: list[str]) -> list[str]:
    """Replace a traveller's saved preferences. An empty list forgets them all."""
    with session_scope() as session:
        profile = session.get(Profile, user_id)
        if profile is None:
            profile = Profile(user_id=user_id, preferences=[])
            session.add(profile)
        profile.preferences = list(preferences)
        return list(profile.preferences)
