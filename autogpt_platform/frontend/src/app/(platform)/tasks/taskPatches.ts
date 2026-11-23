import type { Task } from "@/app/api/__generated__/models/task";
import { TaskStatus } from "@/app/api/__generated__/models/taskStatus";
import type { TaskUpdateRequest } from "@/app/api/__generated__/models/taskUpdateRequest";
import type { TaskPatch } from "./taskCache";

/** How each action changes a task, shown before the server answers. */

export function updatePatch(data: TaskUpdateRequest): TaskPatch {
  return (task) => ({
    ...task,
    ...(data.title ? { title: data.title } : {}),
    ...(data.status ? { status: data.status } : {}),
    ...("start_at" in data
      ? { start_at: data.start_at ?? null, missed_start: false }
      : {}),
  });
}

function withStatus(status: TaskStatus): TaskPatch {
  return (task) => ({ ...task, status });
}

export const startedPatch = withStatus(TaskStatus.IN_PROGRESS);
export const cancelledPatch = withStatus(TaskStatus.CANCELLED);
/** A repeating task's run stops; the task waits for its next run. */
export const runStoppedPatch = withStatus(TaskStatus.TODO);

export const archivedPatch: TaskPatch = () => null;

export function pausedPatch(paused: boolean): TaskPatch {
  return (task: Task) =>
    task.repeat ? { ...task, repeat: { ...task.repeat, paused } } : task;
}

export const notRepeatingPatch: TaskPatch = (task) => ({
  ...task,
  repeat: null,
});
