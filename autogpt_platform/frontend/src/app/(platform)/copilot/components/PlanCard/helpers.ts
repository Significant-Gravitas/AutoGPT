/**
 * Shape of the `data-plan` UI-message part emitted by the backend baseline
 * path when the two-phase planner/executor split is active. Every event of
 * a turn shares one `id`, so the AI SDK reconciles them into a single part
 * that updates in place as the plan progresses.
 */
export type PlanPhase =
  | "planning"
  | "planned"
  | "replanned"
  | "replan_capped"
  | "replan_failed"
  | "skipped"
  | "failed";

export interface PlanStep {
  id: string;
  description: string;
  expectedTools: string[];
  successCriteria: string;
}

export interface PlanPartData {
  phase: PlanPhase;
  steps: PlanStep[];
  plannerModel: string | null;
  executorModel: string | null;
  revision: number;
  reason: string | null;
  executorPrompt: string | null;
}

const PHASE_LABELS: Record<PlanPhase, string> = {
  planning: "Planning your task…",
  planned: "Task plan",
  replanned: "Plan revised",
  replan_capped: "Plan revision limit reached",
  replan_failed: "Plan revision failed",
  skipped: "Answered directly",
  failed: "Planner unavailable",
};

export function getPhaseLabel(data: PlanPartData): string {
  if (data.phase === "planned" && data.steps.length > 0) {
    const count = data.steps.length;
    return `Task plan · ${count} step${count === 1 ? "" : "s"}`;
  }
  if (data.phase === "replanned" && data.revision > 0) {
    return `Plan revised (v${data.revision + 1})`;
  }
  return PHASE_LABELS[data.phase];
}

/** True while the planner call is still in flight (spinner state). */
export function isPlanningInFlight(data: PlanPartData): boolean {
  return data.phase === "planning";
}

/** Strip the model provider prefix for a compact "opus-4.7 → sonnet-4-6" label. */
export function shortModelName(model: string | null): string | null {
  if (!model) return null;
  const slash = model.lastIndexOf("/");
  return slash === -1 ? model : model.slice(slash + 1);
}

/** Narrow an unknown UI-message part payload into `PlanPartData`. */
export function parsePlanPartData(value: unknown): PlanPartData | null {
  if (value == null || typeof value !== "object") return null;
  const raw = value as Record<string, unknown>;
  const phase = raw.phase;
  if (typeof phase !== "string") return null;

  const steps: PlanStep[] = Array.isArray(raw.steps)
    ? raw.steps.flatMap((s) => {
        if (s == null || typeof s !== "object") return [];
        const step = s as Record<string, unknown>;
        return [
          {
            id: typeof step.id === "string" ? step.id : "",
            description:
              typeof step.description === "string" ? step.description : "",
            expectedTools: Array.isArray(step.expectedTools)
              ? step.expectedTools.filter(
                  (t): t is string => typeof t === "string",
                )
              : [],
            successCriteria:
              typeof step.successCriteria === "string"
                ? step.successCriteria
                : "",
          },
        ];
      })
    : [];

  return {
    phase: phase as PlanPhase,
    steps,
    plannerModel:
      typeof raw.plannerModel === "string" ? raw.plannerModel : null,
    executorModel:
      typeof raw.executorModel === "string" ? raw.executorModel : null,
    revision: typeof raw.revision === "number" ? raw.revision : 0,
    reason: typeof raw.reason === "string" ? raw.reason : null,
    executorPrompt:
      typeof raw.executorPrompt === "string" ? raw.executorPrompt : null,
  };
}
