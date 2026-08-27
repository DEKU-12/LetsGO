/** Events the server sends over /ws/plan. */

export interface RouteStep {
  agent: string;
  label: string;
}

export interface PlanResult {
  type: "done" | "clarification";
  trip_id: number | null;
  plan: string | null;
  clarification: string | null;
  destination: string | null;
  route: string[];
  trace: string[];
  sources: string[];
  notes: string[];
  duration_s?: number;
}

export type ServerEvent =
  | { type: "started"; provider: string; model: string; mock: boolean }
  | { type: "route"; plan: RouteStep[]; reasoning: string }
  | { type: "agent"; agent: string; label: string; notes: string[] }
  | { type: "error"; message: string }
  | PlanResult;

export interface Health {
  status: string;
  provider: string;
  model: string;
  mock: boolean;
  agents: string[];
}
