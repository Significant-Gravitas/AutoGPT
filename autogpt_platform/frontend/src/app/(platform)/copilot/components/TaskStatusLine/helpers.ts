import { TaskStatus } from "@/app/api/__generated__/models/taskStatus";
import { asObject, str } from "../ToolChain/resultHelpers";

export interface StartedTask {
  id: string;
  title: string;
  status: TaskStatus;
  ownerExpertID: string | null;
  startedAt: string | null;
  /** The chat doing the work; absent when the task runs in this chat. */
  link: string | null;
}

export type TaskLineState =
  | "working"
  | "needs_input"
  | "done"
  | "failed"
  | "stopped";

const STATES: Record<TaskStatus, TaskLineState> = {
  TODO: "working",
  IN_PROGRESS: "working",
  NEEDS_INPUT: "needs_input",
  DONE: "done",
  FAILED: "failed",
  CANCELLED: "stopped",
};

/** The task a `start_task` call started, read from its result. */
export function startedTaskOf(output: unknown): StartedTask | null {
  const data = asObject(output);
  if (!data || data.type !== "task_started") return null;
  const task = asObject(data.task);
  const id = task && str(task, "id");
  const title = task && str(task, "title");
  if (!task || !id || !title) return null;
  const status = str(task, "status");
  return {
    id,
    title,
    status: isTaskStatus(status) ? status : TaskStatus.IN_PROGRESS,
    ownerExpertID: str(task, "owner_expert_id"),
    startedAt: str(task, "started_at"),
    link: str(data, "worker_session_link"),
  };
}

export function taskLineStateOf(status: TaskStatus): TaskLineState {
  return STATES[status];
}

export function isSettled(status: TaskStatus): boolean {
  const state = STATES[status];
  return state === "done" || state === "failed" || state === "stopped";
}

/** Seconds between two instants, never negative; null when either is unknown. */
export function secondsBetween(
  from: string | Date | null | undefined,
  to: string | Date | number | null | undefined,
): number | null {
  if (!from || !to) return null;
  const start = new Date(from).getTime();
  const end = new Date(to).getTime();
  if (Number.isNaN(start) || Number.isNaN(end)) return null;
  return Math.max(0, Math.round((end - start) / 1000));
}

/** A running clock: 1:24, or 1:02:05 past the hour. */
export function clock(seconds: number): string {
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = String(seconds % 60).padStart(2, "0");
  return h > 0 ? `${h}:${String(m).padStart(2, "0")}:${s}` : `${m}:${s}`;
}

/** How long finished work took: 45s, 4m 12s, 1h 3m. */
export function duration(seconds: number): string {
  if (seconds < 60) return `${seconds}s`;
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  return h > 0 ? `${h}h ${m}m` : `${m}m ${seconds % 60}s`;
}

function isTaskStatus(value: string | null): value is TaskStatus {
  return !!value && value in STATES;
}
