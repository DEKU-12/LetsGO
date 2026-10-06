# LetsGO — a multi-agent travel planner you can actually measure

Give it a request in plain English — *"5 days in Japan, mid-range budget, love
food and history"* — and a supervisor agent routes the work across a team of
specialists, which build a day-by-day travel plan together.

The part that matters: **a first-class evaluation layer**. The system reports
whether the supervisor routed to the right agents, whether the final plans are
any good, and — critically — **how far the LLM judge agrees with a human**.

> Status: phases 1–5 done. Agents, evaluation, API and UI all run.
> Remaining: LangSmith tracing and final polish.

## Run it with zero API keys

One command, one process, browser at http://localhost:8000:

```bash
uv sync && npm --prefix frontend install && npm --prefix frontend run build && uv run uvicorn backend.api:app
```

Or from the terminal, no UI:

```bash
uv run python -m scripts.run_once "5 days in Japan, mid-range budget, food and history"
```

For frontend development, run the API and the Vite dev server separately:

```bash
uv run uvicorn backend.api:app --reload        # :8000
npm --prefix frontend run dev                  # :5173, proxies to :8000
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

19 labeled requests, on two model backends. The columns are separate runs on
different models, months apart — read each on its own, not as a comparison.

| Metric | `claude-sonnet-5` (Aug 2026) | Groq `gpt-oss-120b` (Oct 2026) |
| --- | --- | --- |
| Agent-selection exact match | **19/19 (100%)** | **19/19 (100%)** |
| Micro precision / recall / F1 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 |
| Supervisor plans needing repair | 0% | 5.3% (1/19) |
| Trajectory checks clean | 3/3 | 5/5 |
| Schedules sent back by the itinerary check | — (check added later) | 1/5 |
| Mean plan quality (LLM judge) | 3.6 / 5 | 4.6 / 5 |
| Recommended places confirmed against a map dataset | 65% | 75% (a floor — see below) |
| **Judge validated against human grades** | **not yet** | **not yet — see below** |

The Groq judge scores its own model's plans, and it has not been checked
against a human, so its 4.6 is not evidence that Groq plans are better than
Claude's. Two back-to-back Groq runs gave 4.4 and 4.6, and 82% and 75% of places
confirmed: differences that size are run-to-run noise. The Claude column predates
the country-wide place search described below.

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

**Places are verified, not trusted.** Every attraction the model recommends is
checked against Geoapify — does a place by that name exist at that destination?
The verifier itself is benchmarked (`uv run python -m eval.places_benchmark`):
on 31 labelled cases across two cities and two countries it confirmed 21/21 real
places and rejected 10/10 invented ones. Country-level requests ("5 days in
Japan") used to confirm almost nothing, because the search was a 60 km circle
around the country's midpoint; they are now searched country-wide. Precision is the number that matters there: a false positive would launder
a hallucination as verified fact.

The 65% in the table is a **lower bound**, not an estimate of how often the model
invents places. The current misses are all real — the Acropolis of Athens, Museu
Nacional do Azulejo — that the map dataset did not match. "Unconfirmed" means
"could not confirm", and nothing user-facing calls an unconfirmed place fake.

Geoapify is used as a *verifier* rather than a source on purpose, and the reason
was measured: a radius search around Kyoto returns commemorative plaques and the
city hall, while Fushimi Inari and Kinkaku-ji do not appear at all. OpenStreetMap
knows what is near a point, not what is worth seeing. Sourcing attractions from
it would have made the plans worse.

**The judge is not yet validated.** Its scores (3.6 and 4.6 above) are unverified numbers,
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
| `eval/places_benchmark.py` | Error rate of the place verifier itself |
| `eval/grade.py` | Grade plans by hand, to validate the judge |
| `eval/edits.py` | Plan edits: right section and days, change made, other days untouched |
| `backend/edit.py` | Change a finished plan by chatting: reruns one agent, keeps unmentioned days as they were. Trip-day help: checks live weather and rearranges today if it is bad |
| `backend/api.py` | FastAPI app: `/api/plan`, `/api/trips`, `/api/trips/{id}/edit`, `/api/trips/{id}/today`, `/ws/plan` |
| `backend/db.py` | Trips and per-agent run records |
| `frontend/` | React (Vite) chat UI |

## Design notes

- **The aggregator is not a routing decision.** The supervisor chooses among the
  four workers; the aggregator always runs last to format whatever they produced.
  Routing metrics are computed over the four workers only.
- **The final plan is assembled deterministically** from structured state, so
  nothing reaches the user that an agent did not put into state first. The model
  writes only the opening paragraph, from facts already present.
- **Advice, not fake listings.** Real hotel and flight inventory needs a
  commercial agreement, so the plan does not pretend to have it. "Where to
  stay" recommends neighbourhoods (checked on a map, like attractions) and
  "Getting there and around" gives practical guidance — no invented hotel
  names, prices or fares. Every run ends with the provenance of each data
  source.
- **Agents choose, they do not invent.** The itinerary may only schedule
  attractions the research agent found. Places the model names — attractions
  and neighbourhoods — are verified against a map dataset.
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
