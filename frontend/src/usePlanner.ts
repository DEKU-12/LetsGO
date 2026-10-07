import { useCallback, useRef, useState } from "react";
import type { EditResponse, PlanResult, RouteStep, ServerEvent } from "./types";

/**
 * An anonymous id for this browser, so saved preferences can be found again.
 * There are no accounts; clearing site data simply starts a new profile.
 */
export const USER_ID: string | null = (() => {
  try {
    let id = localStorage.getItem("letsgo-user");
    if (!id) {
      id = crypto.randomUUID();
      localStorage.setItem("letsgo-user", id);
    }
    return id;
  } catch {
    return null; // storage blocked: plan without remembering anything
  }
})();

/** The visitor's model choice and own API key (deployed mode only). */
export interface Credentials {
  provider: string;
  api_key: string;
}

const CREDENTIALS = "letsgo-credentials";

/**
 * Kept in sessionStorage: gone when the tab closes, and only ever sent to this
 * app's own server, with the visitor's own requests. Null when not set (local
 * development, where the server uses its own key).
 */
export function loadCredentials(): Credentials | null {
  try {
    return JSON.parse(sessionStorage.getItem(CREDENTIALS) ?? "null");
  } catch {
    return null;
  }
}

export function saveCredentials(credentials: Credentials): void {
  try {
    sessionStorage.setItem(CREDENTIALS, JSON.stringify(credentials));
  } catch {
    // storage blocked: the choice lasts until the page reloads
  }
}

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

    ws.onopen = () =>
      ws.send(JSON.stringify({ request, user_id: USER_ID, ...loadCredentials() }));

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

        case "suggestions":
          setState((s) =>
            s.result
              ? { ...s, result: { ...s.result, suggested_preferences: message.preferences } }
              : s,
          );
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

  /**
   * Send a change to the server: a typed edit, or "check today". Either way
   * the server saves a new version, or replies without changing anything.
   */
  const send = useCallback(async (url: string, body: object) => {
    setState((s) => ({ ...s, editing: true, reply: null, error: null }));
    try {
      const response = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ...body, ...loadCredentials() }),
      });
      if (!response.ok) {
        const detail = await response.json().catch(() => ({}));
        throw new Error(detail.detail ?? `request failed (${response.status})`);
      }
      const result: EditResponse = await response.json();
      setState((s) =>
        result.type === "reply"
          ? { ...s, editing: false, reply: result.reply }
          : {
              ...s,
              editing: false,
              history: s.result ? [...s.history, s.result] : s.history,
              result,
            },
      );
    } catch (err) {
      setState((s) => ({ ...s, editing: false, error: (err as Error).message }));
    }
  }, []);

  /** Change the current plan. `today` is the trip day the traveller is on, if set. */
  const edit = useCallback(
    (message: string, tripId: number, today: number | null) =>
      send(`/api/trips/${tripId}/edit`, { message, today }),
    [send],
  );

  /** Check today's weather; the server rearranges today only if it is bad. */
  const checkToday = useCallback(
    (tripId: number, today: number) => send(`/api/trips/${tripId}/today`, { today }),
    [send],
  );

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

  return { ...state, plan, edit, checkToday, undo, reset };
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
