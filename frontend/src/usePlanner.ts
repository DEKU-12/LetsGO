import { useCallback, useRef, useState } from "react";
import type { EditResponse, PlanResult, RouteStep, ServerEvent } from "./types";

/** One entry in the live progress list. */
export interface Step {
  agent: string;
  label: string;
  status: "running" | "done";
  notes: string[];
}

export interface PlannerState {
  status: "idle" | "planning" | "done" | "error";
  backend: string | null;
  mock: boolean;
  reasoning: string;
  steps: Step[];
  result: PlanResult | null;
  error: string | null;
  /** An edit request is in flight. */
  editing: boolean;
  /** The server's answer when a message could not be applied as an edit. */
  reply: string | null;
  /** Earlier versions of the plan, most recent last. Undo pops from here. */
  history: PlanResult[];
}

const INITIAL: PlannerState = {
  status: "idle",
  backend: null,
  mock: false,
  reasoning: "",
  steps: [],
  result: null,
  error: null,
  editing: false,
  reply: null,
  history: [],
};

/**
 * Drives one planning run over the WebSocket.
 *
 * The server sends `route` once the supervisor has decided, then one `agent`
 * event per finished node. We seed the step list from the route so the user can
 * see what is coming, not only what has happened.
 */
export function usePlanner() {
  const [state, setState] = useState<PlannerState>(INITIAL);
  const socket = useRef<WebSocket | null>(null);

  const plan = useCallback((request: string) => {
    socket.current?.close();
    setState({ ...INITIAL, status: "planning" });

    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const ws = new WebSocket(`${protocol}//${window.location.host}/ws/plan`);
    socket.current = ws;

    ws.onopen = () => ws.send(JSON.stringify({ request }));

    ws.onmessage = (event) => {
      const message: ServerEvent = JSON.parse(event.data);

      switch (message.type) {
        case "started":
          setState((s) => ({
            ...s,
            backend: `${message.provider} / ${message.model}`,
            mock: message.mock,
          }));
          break;

        case "route":
          setState((s) => ({
            ...s,
            reasoning: message.reasoning,
            steps: mergeRoute(s.steps, message.plan),
          }));
          break;

        case "agent":
          setState((s) => ({ ...s, steps: markDone(s.steps, message) }));
          break;

        case "error":
          setState((s) => ({ ...s, status: "error", error: message.message }));
          ws.close();
          break;

        default:
          setState((s) => ({
            ...s,
            status: "done",
            result: message,
            steps: s.steps.map((step) => ({ ...step, status: "done" })),
          }));
      }
    };

    ws.onerror = () =>
      setState((s) =>
        s.status === "done"
          ? s
          : { ...s, status: "error", error: "lost connection to the server" },
      );
  }, []);

  /** Change the current plan; the server saves the result as a new version. */
  const edit = useCallback(async (message: string, tripId: number) => {
    setState((s) => ({ ...s, editing: true, reply: null, error: null }));
    try {
      const response = await fetch(`/api/trips/${tripId}/edit`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message }),
      });
      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        throw new Error(body.detail ?? `edit failed (${response.status})`);
      }
      const body: EditResponse = await response.json();
      setState((s) =>
        body.type === "reply"
          ? { ...s, editing: false, reply: body.reply }
          : {
              ...s,
              editing: false,
              history: s.result ? [...s.history, s.result] : s.history,
              result: body,
            },
      );
    } catch (err) {
      setState((s) => ({ ...s, editing: false, error: (err as Error).message }));
    }
  }, []);

  /** Go back to the previous version. It is still saved; nothing is deleted. */
  const undo = useCallback(() => {
    setState((s) =>
      s.history.length
        ? { ...s, result: s.history[s.history.length - 1], history: s.history.slice(0, -1), reply: null }
        : s,
    );
  }, []);

  const reset = useCallback(() => {
    socket.current?.close();
    setState(INITIAL);
  }, []);

  return { ...state, plan, edit, undo, reset };
}

/** Add the agents the supervisor chose, keeping anything already finished. */
function mergeRoute(steps: Step[], route: RouteStep[]): Step[] {
  const existing = new Map(steps.map((s) => [s.agent, s]));
  const planned: Step[] = route.map(
    (r) => existing.get(r.agent) ?? { ...r, status: "running", notes: [] },
  );
  const before = steps.filter((s) => !route.some((r) => r.agent === s.agent));
  return [...before, ...planned];
}

function markDone(
  steps: Step[],
  message: { agent: string; label: string; notes: string[] },
): Step[] {
  const known = steps.some((s) => s.agent === message.agent);
  const updated = steps.map((s) =>
    s.agent === message.agent
      ? { ...s, status: "done" as const, notes: message.notes }
      : s,
  );
  if (known) return updated;
  // A step the supervisor did not plan (the schedule check) goes right after
  // the last finished step, which is where it actually ran.
  const at = updated.map((s) => s.status).lastIndexOf("done") + 1;
  return [...updated.slice(0, at), { ...message, status: "done" as const }, ...updated.slice(at)];
}
