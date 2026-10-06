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
  /** How many days the schedule has; 0 when there is no schedule. */
  days: number;
  duration_s?: number;
  /** Set on a version made by an edit: the trip it was edited from. */
  parent_id?: number;
  edited?: {
    section: string;
    days: number[];
    weather?: Weather | null;
    /** False when the agent reran but kept that part of the plan as it was. */
    changed: boolean;
  };
}

export interface Weather {
  description: string | null;
  temp_c: number | null;
  bad: boolean;
}

/** What POST /api/trips/{id}/edit returns. */
export type EditResponse =
  | (PlanResult & { type: "done" })
  | { type: "reply"; reply: string; trip_id: number };

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
