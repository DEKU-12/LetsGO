# ✈️ LetsGO

```
┌──────────────────────────────────────────────────────────────┬────────────────────┐
│  BOARDING PASS · AI TRAVEL PLANNER                           │  CLASS             │
│                                                              │  Free demo         │
│  FROM  one sentence  ─ ─ ─ ✈ ─ ─ ─  TO  a whole trip         │  or your own key   │
│                                                              │                    │
│  CREW  6 AI agents · Claude Opus 5.5                         │  ║║│║║││║│║║│║║│   │
└──────────────────────────────────────────────────────────────┴────────────────────┘
```

**Type one sentence. Get a day-by-day trip — with every place checked on a real
map, a real photo of it, and a schedule that respects opening hours.**

### 🎬 The 25-second trailer

[![Watch the LetsGO trailer](docs/letsgo-trailer.jpg)](https://github.com/DEKU-12/LetsGO/raw/main/docs/letsgo-trailer.mp4)

*Recorded from the real app on Claude Opus: real typing, real agents, real
result. Click to play.*

### 🛫 Try it now → **[letsgo-s9p1.onrender.com](https://letsgo-s9p1.onrender.com)**

Pick **Demo** to fly free (a stand-in model, but real weather, maps and
photos), or paste your own Anthropic or Groq key for the real thing. Your key
stays in your browser tab and is never stored. *Free hosting naps after 15
quiet minutes, so the first load can take about a minute — it's taxiing.*

---

## 🗺️ What happens after you press "Plan it"

You type *"4 days in Kyoto, love temples, gardens and food"*. Then a small
crew gets to work, and you watch each of them finish live:

```
your sentence
   ↓
🛂 Passport control ── parse_request: turns your words into trip details
   │                   (or asks a question if it can't)
   ↓
🧭 The supervisor ──── decides which specialists this trip needs
   │
   ├─▶ 🔎 Researcher ─── finds what's worth seeing, checks the weather for
   │                     each city it chooses (its own tool calls)
   ├─▶ 📅 Scheduler ──── builds the day-by-day plan
   │      └─▶ ✅ Inspector ── checks it: too much in one day? the same place
   │                          twice? stops 300 km apart? a temple at 6 pm
   │                          when it shuts at 5? → sends it back to fix
   ├─▶ 🏨 Stays ──────── which neighbourhoods suit you    ┐ run at the
   └─▶ 🚆 Transport ──── how to arrive and get around     ┘ same time
   ↓
✍️ The writer ──────── assembles the plan from what the crew found
   ↓
🧾 Final check ─────── is the plan complete and well-formed?
```

Six AI agents (supervisor, four specialists, writer), an AI front desk that
reads your request, and two inspectors written in plain code — because a model
grading its own homework tends to give itself an A.

## 🎒 What's in the bag

| Feature | What it does |
| --- | --- |
| 🗺️ **Verified places** | Every attraction and neighbourhood the AI suggests is looked up on a real map (Geoapify). It recommends, the map confirms. |
| 📸 **Real photos** | Each place gets *its own* photo — found through the same map entry that verified it, so "Gion" can't come back with a photo of a different Gion. Credited, Wikimedia-licensed. No photo of the place? You get a photo of the city, clearly labelled as one. |
| 🕘 **Opening hours** | Read from the map and given to the scheduler, so Kiyomizu-dera lands at its 06:00 opening, not at 19:00 when it's shut. |
| 🌦️ **Weather per city** | A Tokyo-and-Kyoto trip gets the forecast for both — the AI decides which cities to check. |
| 💬 **Edit by chatting** | "Less walking on day 3." Only day 3 changes; every other day stays exactly as it was. Undo is one click. |
| ☔ **Trip-day help** | On the trip? Pick your day, press *Check today's weather*. If it's raining, today gets rearranged around indoor places. Sunny? It says so and leaves it alone. |
| 🧠 **Remembers you (if you say yes)** | "I'm vegetarian" gets offered as something to remember — never saved without a click — and every later plan uses it. "It's for my mum's birthday" is never offered. |
| 🏨 **Advice, not fake listings** | No invented hotels or fares. You get the neighbourhoods worth staying in and how to get around — the parts a well-travelled friend could actually tell you. |
| 🔑 **Bring your own key** | The public site uses *your* key for *your* request, then forgets it. |

## 📊 The flight recorder (evaluation)

Most demos show you a nice plan. This one shows you its numbers — measured on
**Claude Opus 5.5** (Oct 2026), the model it runs on.

| What we measured | Result |
| --- | --- |
| **Supervisor picks the right specialists** (31 labelled requests, incl. 12 tricky ones) | **30/31** (96.8%) |
| **…on every one of 3 repeat runs** (pass^3) | **30/31** — the one miss is consistent, not random |
| **Ignores prompt injection** ("ignore all previous instructions…", fake "admin mode") | **2/2** |
| Precision / recall / F1 of agent choice | 1.00 / 0.98 / 0.99 |
| Supervisor plans the code had to repair | 2/31 |
| **Researcher checks weather for the right cities** | **10/10** (Rome, Florence *and* Venice included) |
| …right on all 3 repeat runs (pass^3) | **7/7** requests |
| Invalid or duplicate tool calls | **0** of 26 |
| Times code had to fetch weather the AI skipped | 0/7 |
| Agent paths clean (right order, nothing wasted) | 5/5 |
| Recommended places confirmed on a real map | 77% — a *floor*, see below |
| Schedules the inspector had to send back | 1 of 5 plans |
| Plan quality (AI judge, 1–5) | 3.5 — **not yet validated by a human**, see below |
| ⏱️ Time per plan | **61 s** average (99 s slowest), down from 78 s |
| 💸 Cost per plan | **~$0.12** (~12k tokens) |

And two checks where no AI is involved at all:

| What we measured | Result |
| --- | --- |
| **Place checker accuracy** (31 labelled places, 10 of them invented) | **21/21** real places confirmed, **10/10** fakes rejected |
| **Photos** (21 real places, incl. names straight from a real Miami plan) | 19 with a real photo, 18 of the place itself, **0 wrong** |

### Reading the numbers honestly

- **The 30/31 miss might be my label, not the AI.** For *"two weeks backpacking
  Vietnam, north to south by bus"* I expected it to plan where to stay; Opus
  decided I hadn't asked. Fair point, honestly.
- **77% of places confirmed is a floor, not a hallucination rate.** The misses
  so far are real places the map lists in the local language — the Acropolis,
  Museu Nacional do Azulejo. "Unconfirmed" means "couldn't confirm", and the app
  never calls one fake. What matters is the other direction: the checker has
  never confirmed a made-up place (10/10 rejected).
- **The judge's 3.5 is unverified.** An AI grading plans needs to be checked
  against a human first. Until someone grades a batch by hand
  (`uv run python -m eval.grade`), treat it as a number, not a verdict.
- **The eval layer pays for itself.** Its very first run caught a bug that had
  passed every unit test: questions like *"what's the weather in Rome?"* were
  being rejected as "not a travel request". Later, the speed measurements
  caught the inspector raising false alarms (it thought Ginkaku-ji was a repeat
  of Kinkaku-ji). Fixing that, plus two speed-ups, cut the slowest plan from
  183 s to 99 s.

### Run the scorecards yourself

```bash
uv run python -m eval.run --repeat 3      # routing (pass^3), plans, judge, time, cost
uv run python -m eval.tools --repeat 3    # the researcher's tool calls
uv run python -m eval.places_benchmark    # the place checker itself
uv run python -m eval.photos              # photos and opening hours
uv run python -m eval.edits               # chat edits: right days changed, others untouched
uv run python -m eval.preferences         # remembers "vegetarian", never "mum's birthday"
uv run python -m eval.grade               # you grade plans, to check the AI judge
```

Add `--provider mock` to any of them to check the plumbing for free.

## 🛠️ Engine room

```
LangGraph (agents)  ·  FastAPI + WebSocket (live progress)  ·  React + Vite (UI)
SQLAlchemy + SQLite  ·  Claude Opus 5.5 / Groq / mock  ·  LangSmith (traces, tokens, cost)
Geoapify (maps)  ·  OpenWeather  ·  Wikidata + Wikimedia Commons (photos)
```

| Where | What lives there |
| --- | --- |
| `backend/graph.py` | The flight plan: how the agents connect (stays and transport run in parallel) |
| `backend/agents/` | The crew: supervisor, researcher, scheduler, inspector, stays, transport, writer |
| `backend/llm.py` | One wrapper for every model call: Claude, Groq or mock; tool calling; token and cost tracking |
| `backend/adapters/` | Maps, weather, photos — each with a mock mode, so it all runs with zero keys |
| `backend/hours.py` | Reads opening hours like `Mo-Fr 09:00-17:00; Su off` — and refuses to guess at anything stranger |
| `backend/edit.py` | Chat edits and trip-day help |
| `backend/preferences.py` | Spots lasting preferences to offer remembering |
| `backend/api.py` | The API and live progress stream; each trip belongs to the browser that made it |
| `eval/` | Every scorecard above |
| `frontend/` | The chat-style UI |

## 🚀 Run it yourself

**Zero keys** — everything has a stand-in, so a fresh clone just works:

```bash
uv sync && npm --prefix frontend install && npm --prefix frontend run build && uv run uvicorn backend.api:app
```

Then open http://localhost:8000. Or skip the UI:

```bash
uv run python -m scripts.run_once "5 days in Japan, mid-range budget, food and history"
```

**For real plans**, copy `.env.example` to `.env` and add an `ANTHROPIC_API_KEY`
(or a free `GROQ_API_KEY`). Add `GEOAPIFY_API_KEY` and `OPENWEATHER_API_KEY`
(both free) for real maps, photos and weather.

**Deploy your own:** the repo ships a `render.yaml`. On Render, *New →
Blueprint*, pick this repo, paste your two map/weather keys. It runs with
`REQUIRE_USER_KEY=true`, so visitors bring their own model key and your bill
stays at zero.

## 🦺 Safety on board

- **Your key, your request, then gone.** Used for that request only; never
  saved, logged or traced. With `REQUIRE_USER_KEY=true` the server never falls
  back to the owner's key.
- **Your trips are yours.** Each trip belongs to the browser that made it —
  nobody else can list, read or edit it, even by guessing trip numbers.
- **The plan can't run code.** It's built from AI output, so it's sanitised
  (DOMPurify) before it hits a page that might hold your key.
- **Preferences are opt-in** and deletable, one tap each.

## 🌩️ Known turbulence

- **Whole-country trips** ("5 days in Japan") sometimes match a same-named
  place in the wrong city — usually costing one extra schedule check.
- **The scheduler is told** to use only researched places, but not forced to; the
  scorecard lists any stop it slips in (a lane walk here, a promenade there).
- **Visa, safety and transport tips** come from the model and aren't checked
  against official sources — the plan says so, right under them.
- **Demo mode's stand-in model** is a puppet: real maps and photos, but canned
  attractions. The trailer above is the real thing.

## 🧑‍✈️ Ground crew

- **144 tests** — `uv run pytest`. They run with no keys and touch no paid API.
- **CI on every push** (GitHub Actions): the tests, every scorecard on the stand-in
  model, and the frontend build.
- **CD to Render** — a push to `main` deploys only after CI passes.
- **LangSmith traces** — every agent step, tool call, token and cent, when
  `LANGSMITH_TRACING=true`.

*Bon voyage.* ✈️
