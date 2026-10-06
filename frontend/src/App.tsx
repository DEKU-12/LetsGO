import { useEffect, useMemo, useState } from "react";
import { marked } from "marked";
import { usePlanner } from "./usePlanner";
import type { Health, PlanResult } from "./types";

const EDIT_EXAMPLES = [
  "Less walking on day 2",
  "Nothing before 10am",
  "A cheaper place to stay",
];

const EXAMPLES = [
  "5 days in Japan, mid-range budget, love food and history. I need hotels and flights too.",
  "What's the weather like in Rome in October?",
  "Plan 4 days in Porto with somewhere to stay. I'm driving there myself.",
];

export default function App() {
  const planner = usePlanner();
  const [request, setRequest] = useState("");
  const [health, setHealth] = useState<Health | null>(null);

  useEffect(() => {
    fetch("/api/health")
      .then((r) => r.json())
      .then(setHealth)
      .catch(() => setHealth(null));
  }, []);

  const busy = planner.status === "planning" || planner.editing;

  function submit(event: React.FormEvent) {
    event.preventDefault();
    const text = request.trim();
    if (text && !busy) planner.plan(text);
  }

  return (
    <div className="app">
      <header>
        <h1>
          Lets<span>GO</span>
        </h1>
        <p className="tagline">
          A supervisor routes your request across specialist agents, then they
          build a plan together.
        </p>
        {health && (
          <p className="backend">
            {health.provider} / {health.model}
            {health.mock && <span className="warn"> · mock data, no API key set</span>}
          </p>
        )}
      </header>

      <form onSubmit={submit}>
        <textarea
          value={request}
          onChange={(e) => setRequest(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) submit(e);
          }}
          placeholder="Where do you want to go? Tell me how long, roughly what budget, and what you enjoy."
          rows={3}
          disabled={busy}
        />
        <div className="actions">
          <div className="examples">
            {EXAMPLES.map((example) => (
              <button
                key={example}
                type="button"
                className="chip"
                disabled={busy}
                onClick={() => setRequest(example)}
              >
                {example.length > 42 ? example.slice(0, 40) + "…" : example}
              </button>
            ))}
          </div>
          <button type="submit" className="go" disabled={busy || !request.trim()}>
            {busy ? "Planning…" : "Plan it"}
          </button>
        </div>
      </form>

      {planner.status !== "idle" && (
        <Progress
          steps={planner.steps}
          reasoning={planner.reasoning}
          busy={busy}
        />
      )}

      {planner.error && <p className="error">{planner.error}</p>}

      {planner.result?.clarification && (
        <div className="clarify">
          <strong>One question first:</strong> {planner.result.clarification}
        </div>
      )}

      {planner.result?.plan && planner.result.trip_id !== null && (
        <EditBox
          tripId={planner.result.trip_id}
          result={planner.result}
          editing={planner.editing}
          reply={planner.reply}
          canUndo={planner.history.length > 0}
          onEdit={planner.edit}
          onUndo={planner.undo}
        />
      )}

      {planner.result?.plan && <Plan markdown={planner.result.plan} />}

      {planner.result && (
        <Meta
          sources={planner.result.sources}
          seconds={planner.result.duration_s}
          notes={planner.result.notes}
        />
      )}
    </div>
  );
}

function Progress({
  steps,
  reasoning,
  busy,
}: {
  steps: { agent: string; label: string; status: string }[];
  reasoning: string;
  busy: boolean;
}) {
  return (
    <section className="progress">
      {reasoning && <p className="reasoning">“{reasoning}”</p>}
      <ol>
        {steps.map((step) => (
          <li key={step.agent} className={step.status}>
            <span className="dot" aria-hidden="true" />
            {step.label}
          </li>
        ))}
        {busy && steps.length === 0 && (
          <li className="running">
            <span className="dot" aria-hidden="true" />
            Reading your request
          </li>
        )}
      </ol>
    </section>
  );
}

function EditBox({
  tripId,
  result,
  editing,
  reply,
  canUndo,
  onEdit,
  onUndo,
}: {
  tripId: number;
  result: PlanResult;
  editing: boolean;
  reply: string | null;
  canUndo: boolean;
  onEdit: (message: string, tripId: number) => void;
  onUndo: () => void;
}) {
  const [message, setMessage] = useState("");

  function submit(event: React.FormEvent) {
    event.preventDefault();
    const text = message.trim();
    if (text && !editing) {
      onEdit(text, tripId);
      setMessage("");
    }
  }

  return (
    <form className="edit" onSubmit={submit}>
      <div className="edit-row">
        <input
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          placeholder="Change something… e.g. less walking on day 2"
          disabled={editing}
          aria-label="Change the plan"
        />
        <button type="submit" className="go" disabled={editing || !message.trim()}>
          {editing ? "Updating…" : "Update"}
        </button>
      </div>
      <div className="actions">
        <div className="examples">
          {EDIT_EXAMPLES.map((example) => (
            <button
              key={example}
              type="button"
              className="chip"
              disabled={editing}
              onClick={() => setMessage(example)}
            >
              {example}
            </button>
          ))}
        </div>
        {canUndo && (
          <button type="button" className="chip" disabled={editing} onClick={onUndo}>
            Undo last change
          </button>
        )}
      </div>
      {result.edited && !reply && <p className="edit-note">{describeEdit(result.edited)}</p>}
      {reply && <p className="edit-note">{reply}</p>}
    </form>
  );
}

function describeEdit(edited: { section: string; days: number[] }): string {
  const section =
    { itinerary: "the schedule", accommodation: "where to stay", transport: "transport" }[
      edited.section
    ] ?? edited.section;
  const days = edited.days.length
    ? ` (day${edited.days.length > 1 ? "s" : ""} ${edited.days.join(", ")})`
    : "";
  return `Updated ${section}${days}. Everything else is unchanged.`;
}

function Plan({ markdown }: { markdown: string }) {
  const html = useMemo(() => marked.parse(markdown, { async: false }), [markdown]);
  return (
    <article className="plan" dangerouslySetInnerHTML={{ __html: html as string }} />
  );
}

function Meta({
  sources,
  seconds,
  notes,
}: {
  sources: string[];
  seconds?: number;
  notes: string[];
}) {
  const [open, setOpen] = useState(false);
  if (!sources.length && !notes.length) return null;

  return (
    <footer className="meta">
      <span>
        {seconds !== undefined && `${seconds}s`}
        {sources.length > 0 && ` · ${sources.join(", ")}`}
      </span>
      {notes.length > 0 && (
        <>
          <button type="button" onClick={() => setOpen((v) => !v)}>
            {open ? "hide" : `${notes.length} note${notes.length > 1 ? "s" : ""}`}
          </button>
          {open && (
            <ul>
              {notes.map((note) => (
                <li key={note}>{note}</li>
              ))}
            </ul>
          )}
        </>
      )}
    </footer>
  );
}
