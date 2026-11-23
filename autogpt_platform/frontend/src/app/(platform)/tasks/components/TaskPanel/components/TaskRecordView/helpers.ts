import type { Expert } from "@/app/api/__generated__/models/expert";
import type { Task } from "@/app/api/__generated__/models/task";
import type { TaskActivity } from "@/app/api/__generated__/models/taskActivity";
import type { TaskAsk } from "@/app/api/__generated__/models/taskAsk";

export const ACTIVITY_PREVIEW = 8;

/** Who did it, in the words the panel uses. */
export function actorName(line: TaskActivity, experts: Expert[]) {
  if (line.actor_type === "USER") return "You";
  if (line.actor_type === "OTTO") return "Otto";
  if (line.actor_type === "EXPERT") {
    return (
      experts.find((expert) => expert.id === line.actor_expert_id)?.name ??
      "An expert"
    );
  }
  return null;
}

/** The subtask a line or a question belongs to; null for the task itself. */
export function subtaskTitle(taskID: string, task: Task, subtasks: Task[]) {
  if (taskID === task.id) return null;
  return subtasks.find((subtask) => subtask.id === taskID)?.title ?? null;
}

export function openAsks(asks: TaskAsk[]) {
  return asks.filter((ask) => ask.status === "OPEN");
}

export function answeredAsks(asks: TaskAsk[]) {
  return asks.filter((ask) => ask.status === "ANSWERED" && ask.answer);
}
