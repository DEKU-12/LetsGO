# LetsGO — a multi-agent travel planner you can actually measure

Give it a request in plain English — *"5 days in Japan, mid-range budget, love
food and history"* — and a supervisor agent routes the work across a team of
specialists, which build a day-by-day travel plan together.

The part that matters: **a first-class evaluation layer**. The system reports
whether the supervisor routed to the right agents, whether the final plans are
any good, and — critically — **how far the LLM judge agrees with a human**.

> Status: in progress. Phases 1–3 are done — all five agents run end to end
> and the evaluation layer reports on them. Backend API, UI and live places
> data are still to come.

## Run it with zero API keys

```bash
uv sync
uv run python -m scripts.run_once "5 days in Japan, mid-range budget, food and history"
```

Every external call goes through an adapter with a mock mode, and the model
wrapper itself falls back to deterministic canned responses. With an empty
environment the whole graph still runs end to end — that is the default, not a
degraded path.

To use a real model, copy `.env.example` to `.env` and set `ANTHROPIC_API_KEY`
(the project targets `claude-sonnet-5`).

## Evaluation results

```
uv run python -m eval.run
```

Latest run, `claude-sonnet-5`, 19 labeled requests:

| Metric | Result |
| --- | --- |
| Agent-selection exact match | **19/19 (100%)** |
| Micro precision / recall / F1 | 1.00 / 1.00 / 1.00 |
| Supervisor plans needing repair | 0% |
| Trajectory checks clean | 3/3 |
| Mean plan quality (LLM judge) | 3.5 / 5 |
| **Judge validated against human grades** | **not yet — see below** |

**Read the 100% sceptically.** It does not mean the router is perfect; it means
this 19-case dataset has stopped discriminating. The honest reading is "no known
failure mode in the cases tested so far", and the next thing the dataset needs is
genuinely ambiguous requests, not more easy ones.

The first run of this suite scored 84.2%, and the three failures were all the
same bug: the input guard was rejecting questions like *"what's the weather in
Rome"* as "not a travel request", so the supervisor never saw them. That bug had
been in the code since phase 1, passed every unit test, and was invisible in
manual testing. The eval layer found it on its first run — which is the argument
for building evaluation early rather than last.

**The judge is not yet validated.** Its 3.5/5 is currently an unverified number,
and the report says so rather than presenting it as a result. Validation needs
human-graded plans:

```
uv run python -m eval.grade
```

## Architecture

```
request
  ↓
parse_request (input guard)  ── unusable? ──▶ clarifying question
  ↓
supervisor ⇄ destination_research
           ⇄ itinerary
           ⇄ accommodation
           ⇄ transport
  ↓
aggregator → validate_plan (output guard) → markdown plan
```

The supervisor writes an ordered `route_plan` into shared state and every
worker returns to it, so `state["trace"]` is a real record of the path taken —
which is what the trajectory checks in `eval/` score.

| Piece | What it does |
| --- | --- |
| `backend/state.py` | The typed state every node reads and writes |
| `backend/graph.py` | LangGraph wiring |
| `backend/guards.py` | Free text → validated `TripParams`; output well-formedness |
| `backend/agents/` | Supervisor + worker nodes |
| `backend/adapters/` | External APIs, each with a mock mode |
| `eval/dataset.py` | 19 labeled requests + stored human grades |
| `eval/selection.py` | Per-agent precision/recall, micro average, exact match |
| `eval/judge.py` | LLM-as-judge on a 5-part rubric, plus judge-vs-human validation |
| `eval/trajectory.py` | Dependency order, unused output, repaired routes |
| `eval/run.py` | The report |
| `eval/grade.py` | Grade plans by hand, to validate the judge |

## Design notes

- **The aggregator is not a routing decision.** The supervisor chooses among the
  four workers; the aggregator always runs last to format whatever they produced.
  Routing metrics are computed over the four workers only.
- **The final plan is assembled deterministically** from structured state, so
  nothing reaches the user that an agent did not put into state first. The model
  writes only the opening paragraph, from facts already present.
- **Mock data says it is mock.** Hotels and flights have no live provider —
  real inventory needs a commercial agreement — so those sections of the plan
  carry an explicit line saying the numbers are illustrative, and every run
  ends with the provenance of each data source.
- **Agents choose, they do not invent.** The itinerary may only schedule
  attractions the research agent found; accommodation and transport may only
  pick from the adapter's shortlist. Anything else is dropped and recorded.
- **Model backends are swappable** (`backend/llm.py`): `anthropic` (spec default),
  `groq` (cheap iteration), `mock` (no keys). Selected automatically.

## Build phases

1. ✅ Skeleton: structure, `uv` env, one adapter, supervisor + one worker, runs end to end
2. ✅ All agents + shared state
3. ✅ Eval layer (selection metrics, LLM judge + judge validation, trajectory checks)
4. ⬜ FastAPI backend with WebSocket progress + storage
5. ⬜ React (Vite) chat UI
6. ⬜ Real APIs where keys exist + LangSmith tracing
7. ⬜ Polish: architecture diagram, live demo path, eval results up front

## Tests

```bash
uv run pytest
```
