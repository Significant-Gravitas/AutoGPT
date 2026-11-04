import { toPlan, toTask } from "@/app/(platform)/tasks/data/normalize";
import type { Task, TaskPlan } from "@/app/(platform)/tasks/data/types";

function parseObject(output: unknown): Record<string, unknown> | null {
  if (!output) return null;
  if (typeof output === "string") {
    const trimmed = output.trim();
    if (!trimmed) return null;
    try {
      return parseObject(JSON.parse(trimmed) as unknown);
    } catch {
      return null;
    }
  }
  if (typeof output === "object" && !Array.isArray(output)) {
    return output as Record<string, unknown>;
  }
  return null;
}

export function getProposedPlan(output: unknown): TaskPlan | null {
  const obj = parseObject(output);
  if (!obj || obj.type !== "task_plan") return null;
  return toPlan(obj.plan);
}

export function getConfirmedPlan(
  output: unknown,
): { plan: TaskPlan | null; parent: Task | null } | null {
  const obj = parseObject(output);
  if (!obj || obj.type !== "task_plan_confirmed") return null;
  return {
    plan: toPlan(obj.plan),
    parent: obj.parent_task ? toTask(obj.parent_task) : null,
  };
}

export function getErrorMessage(output: unknown) {
  const obj = parseObject(output);
  if (!obj) return null;
  if (typeof obj.error === "string") return obj.error;
  if (obj.type === "error" && typeof obj.message === "string") {
    return obj.message;
  }
  return null;
}

export function getPlanShapeLabel(plan: TaskPlan) {
  const count = plan.steps.length;
  if (plan.split) return `${count} ${count === 1 ? "sub-task" : "sub-tasks"}`;
  return `One task · ${count} ${count === 1 ? "item" : "items"}`;
}

export function taskHref(id: string) {
  return `/tasks?task=${encodeURIComponent(id)}`;
}

const TASK_PLAN_PART_TYPES = new Set([
  "tool-propose_plan",
  "tool-confirm_task_plan",
]);

/** Plan cards stay on screen as parts of their own, outside the ToolChain. */
export function isTaskPlanPart(part: { type: string }) {
  return TASK_PLAN_PART_TYPES.has(part.type);
}
