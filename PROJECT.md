# PROJECT.md — Travel Planning Multi-Agent System

## What we're building

A multi-agent travel planning system. A user submits a travel request in plain English ("5 days in Japan, mid-range budget, love food and history"), and a team of specialized AI agents — coordinated by a supervisor — research the destination, build a day-by-day itinerary, find accommodation, and plan transport, then combine everything into one coherent travel plan.

The system is built on **LangGraph** using a **supervisor-worker** pattern. What sets this build apart from a typical demo is a **built-in evaluation layer**: we measure whether the supervisor routes to the right agents and whether the final plan is actually good, using tool-selection metrics and LLM-as-judge scoring. Evaluation is a first-class part of the project, not an afterthought.

## Who this is for / why it matters

Portfolio project for a junior AI Engineer job search. It must demonstrate: real multi-agent orchestration, live API integration, clean deployment, and — most importantly — rigorous evaluation of the agent system. The eval layer is the differentiator; do not treat it as optional.

## Tech stack (use exactly these unless a blocker forces a change — flag it if so)

- **Orchestration:** LangGraph (supervisor + worker agents, shared state)
- **LLM:** Anthropic Claude via API (use `claude-sonnet` class model). Keep the model behind a small wrapper so it's swappable.
- **Backend:** FastAPI, with a WebSocket endpoint for streaming progress updates
- **Frontend:** React (Vite) — simple, clean chat-style UI. Keep it minimal; the backend is the star.
- **Storage:** PostgreSQL (via SQLAlchemy) for saved trips and run logs. SQLite is acceptable for local dev.
- **Observability:** LangSmith for tracing agent runs (optional but wire it in if an API key is present)
- **Evaluation:** a dedicated `eval/` module (details below)
- **Python:** 3.11+, managed with `uv`. Every dependency pinned.

## External APIs (all should degrade gracefully to a mock if no key is set)

- **Places / attractions & maps:** Google Places API (or a free alternative like OpenTripMap if simpler)
- **Weather:** OpenWeather API
- **Flights/transport:** use a free/mock source — real flight APIs are pay-gated; a stubbed adapter with realistic fake data is fine and should be clearly labeled
- **Accommodation:** mock adapter with realistic data (real hotel APIs are gated too)
- **News/events:** optional

Every external call goes through an **adapter** with a consistent interface, and every adapter has a **mock mode** so the whole system runs end-to-end with zero API keys. This matters for the demo and for testing.

## The agents

Build these as LangGraph nodes sharing a typed state object.

1. **Supervisor Agent** — reads the user request, decides which worker agents to invoke and in what order, and routes the shared state between them. This is the orchestrator. Its routing decisions are what the eval layer will measure.
2. **Destination Research Agent** — gathers info on the place: top attractions, weather for the dates, any visa/safety notes. Uses Places + Weather adapters.
3. **Itinerary Planning Agent** — builds a day-by-day schedule from the research, respecting trip length and stated preferences.
4. **Accommodation Agent** — finds lodging options within the stated budget.
5. **Transport Booking Agent** — proposes flights/trains/local transport options.
6. **Response Aggregator Agent** — merges all agent outputs into one clean, structured travel plan (markdown) for the user.

## Shared state (LangGraph)

Define a typed state (TypedDict or Pydantic) that carries: the original request, parsed trip parameters (destination, dates, duration, budget, preferences), each agent's outputs, and a running trace of which agents ran. Agents read prior results from state and write their own — they build on each other, they don't start fresh.

## The evaluation layer — THIS IS THE POINT, BUILD IT PROPERLY

Create an `eval/` module with three parts. This is what makes the project stand out, so give it real care.

1. **Tool/agent-selection accuracy.**
   - Create a small labeled dataset (start with ~15–20 example requests) where each request is annotated with the correct set of agents the supervisor *should* invoke (e.g. a "what's the weather in Rome" request needs only Destination Research, not Transport).
   - Run the supervisor on each, compare its actual routing against the labels, and report precision/recall on agent selection plus exact-match on the full routing set.

2. **LLM-as-judge scoring of final plans.**
   - Define a rubric (e.g. completeness, adherence to budget, adherence to stated preferences, coherence of the day-by-day schedule, realism).
   - Use an LLM judge to score each generated plan against the rubric, requiring the judge to give reasoning before a numeric score.
   - **Critically: validate the judge.** Include a small set of human-labeled plans (you'll grade ~10 yourself) and report how well the LLM judge agrees with the human labels. State this agreement number. A judge you haven't checked is not trustworthy.

3. **Trajectory / process check.**
   - For multi-step runs, record the sequence of agents actually invoked and flag runs where the path was invalid or wasteful (e.g. an agent ran but its output was never used).

Expose all of this as a runnable script: `uv run python -m eval.run` that prints a clean report. Store eval results so they can be compared across runs.

## Guardrails

Add a lightweight input/output guard layer:
- Validate/parse the user request into structured trip parameters before agents run; reject or clarify nonsensical requests rather than passing garbage downstream.
- Ensure the aggregator's output is well-formed structured data before returning it.
Keep this simple but present — it's part of the "trustworthy system" story.

## Project structure (suggested)

```
travel-agents/
  backend/
    agents/          # supervisor + worker nodes
    adapters/        # external API adapters, each with mock mode
    graph.py         # LangGraph wiring, shared state definition
    guards.py        # input/output validation
    api.py           # FastAPI app + WebSocket
    db.py            # storage
  eval/
    dataset.py       # labeled requests + human-labeled plans
    selection.py     # agent-selection metrics
    judge.py         # LLM-as-judge + judge validation
    trajectory.py    # process checks
    run.py           # runnable report
  frontend/          # React (Vite) chat UI
  tests/
  README.md
  pyproject.toml
```

## Build order (do it in phases — get each working before moving on)

1. **Skeleton:** project structure, `uv` env, one adapter in mock mode, a trivial LangGraph graph with just the supervisor + one worker, runnable end-to-end from a Python script (no UI yet). Prove the graph runs.
2. **All agents + shared state:** build the four workers and the aggregator, all reading/writing shared state, all adapters in mock mode. End-to-end plan generation from a script.
3. **Eval layer:** build the `eval/` module — this is not optional and not last-minute. Get agent-selection metrics and LLM-as-judge (with judge validation) working against the labeled dataset.
4. **Backend API:** wrap it in FastAPI with the WebSocket progress stream. Add storage.
5. **Frontend:** minimal React chat UI that submits a request and streams progress + shows the final plan.
6. **Real APIs + observability:** swap mock adapters for real ones where keys exist (Places, Weather); wire LangSmith tracing.
7. **Polish:** README with the architecture diagram, a live-demo path, and the eval results printed prominently.

## Definition of done

- Runs end-to-end from a fresh clone with **zero API keys** (all adapters fall back to mock) via one documented command.
- The `eval/` report runs and prints agent-selection metrics, LLM-as-judge scores, **and the judge-vs-human agreement number**.
- README opens with the problem, shows the architecture, explains the agent roles, and — prominently — reports the evaluation results and what they mean.
- Tests cover the graph wiring, the adapters' mock mode, and the eval metrics.
- Clean, typed, documented code. Secrets in `.env`, never committed.

## Guidance for the build

- Keep the LLM model behind a thin wrapper so it's swappable and so eval runs are cheap to configure.
- Every adapter must have a mock mode — the project has to run and demo without paid API keys.
- Write the labeled eval dataset early (phase 3), even small; it's the backbone of the differentiator.
- Prefer a working small version at each phase over a broken ambitious one. Ship phase by phase.
- Don't over-engineer. Match the code to the spec; resist adding frameworks or abstractions the spec doesn't call for.
