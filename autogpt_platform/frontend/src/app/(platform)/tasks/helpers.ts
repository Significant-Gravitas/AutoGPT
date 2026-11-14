import type { Expert } from "@/app/api/__generated__/models/expert";
import type { Task } from "@/app/api/__generated__/models/task";
import { TaskStatus } from "@/app/api/__generated__/models/taskStatus";
import type { Tone } from "@/components/styles/tones";
import {
  AUTOPILOT_AVATAR_URL,
  AUTOPILOT_NAME,
} from "@/components/molecules/AutopilotAvatar/helpers";

/** The lanes every surface sorts a task into: board columns and pills. */
export const LANES = ["needs_you", "working", "done", "cancelled"] as const;
export type Lane = (typeof LANES)[number];

const LANE_LABELS: Record<Lane, string> = {
  needs_you: "Needs you",
  working: "Working",
  done: "Done",
  cancelled: "Cancelled",
};

export interface StatusView {
  lane: Lane;
  /** The one word the pill shows. */
  word: string;
  tone: Tone;
}

const STATUS_VIEWS: Record<TaskStatus, StatusView> = {
  TODO: { lane: "working", word: "To do", tone: "quiet" },
  IN_PROGRESS: { lane: "working", word: "Working", tone: "info" },
  NEEDS_INPUT: { lane: "needs_you", word: "Needs you", tone: "waiting" },
  FAILED: { lane: "needs_you", word: "Failed", tone: "error" },
  DONE: { lane: "done", word: "Done", tone: "ok" },
  CANCELLED: { lane: "cancelled", word: "Cancelled", tone: "quiet" },
};

/** The order the status menu lists them in. */
export const STATUS_ORDER: TaskStatus[] = [
  TaskStatus.TODO,
  TaskStatus.IN_PROGRESS,
  TaskStatus.NEEDS_INPUT,
  TaskStatus.DONE,
  TaskStatus.FAILED,
  TaskStatus.CANCELLED,
];

export function statusOf(status: TaskStatus): StatusView {
  return STATUS_VIEWS[status] ?? STATUS_VIEWS.TODO;
}

export function laneOf(task: Task): Lane {
  return statusOf(task.status).lane;
}

/** Done, failed and cancelled tasks don't change on their own any more. */
export function isSettled(status: TaskStatus) {
  return (
    status === TaskStatus.DONE ||
    status === TaskStatus.FAILED ||
    status === TaskStatus.CANCELLED
  );
}

export const POLL_MS = 5_000;

/** Re-read every few seconds while any task can still move on its own. */
export function pollInterval(tasks: Task[] | undefined) {
  if (!tasks) return false;
  return tasks.some((t) => !isSettled(t.status)) ? POLL_MS : false;
}

export interface BoardColumn {
  id: Lane;
  label: string;
  tasks: Task[];
}

export function buildBoardColumns(tasks: Task[]): BoardColumn[] {
  return LANES.map((id) => ({
    id,
    label: LANE_LABELS[id],
    tasks: tasks.filter((t) => laneOf(t) === id),
  }));
}

export interface TaskOwner {
  name: string;
  avatarUrl: string | null;
}

/** Who works the task: Otto when no expert owns it. */
export function ownerOf(task: Task, experts: Expert[]): TaskOwner {
  if (!task.owner_expert_id) {
    return { name: AUTOPILOT_NAME, avatarUrl: AUTOPILOT_AVATAR_URL };
  }
  const expert = experts.find((e) => e.id === task.owner_expert_id);
  return {
    name: expert?.name ?? "An expert",
    avatarUrl: expert?.avatar_url ?? null,
  };
}

/** The one duration formatter: "20s", "4m", "5h", "2d". */
export function formatAge(ms: number) {
  const seconds = Math.max(0, Math.floor(ms / 1000));
  if (seconds < 60) return `${seconds}s`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h`;
  return `${Math.floor(hours / 24)}d`;
}

function since(at: Date | string | null | undefined, now: number) {
  if (!at) return null;
  const ms = now - new Date(at).getTime();
  return Number.isFinite(ms) ? formatAge(ms) : null;
}

/** How long, worded for the lane: "12m" of work so far, "waiting 5h" on
 *  you, "2d ago" for finished work. */
export function statusAge(task: Task, now: number) {
  switch (task.status) {
    case TaskStatus.IN_PROGRESS:
      return since(task.started_at ?? task.updated_at, now);
    case TaskStatus.NEEDS_INPUT: {
      const age = since(task.updated_at, now);
      return age ? `waiting ${age}` : null;
    }
    case TaskStatus.TODO: {
      const age = since(task.created_at, now);
      return age ? `${age} ago` : null;
    }
    default: {
      const age = since(task.completed_at ?? task.updated_at, now);
      return age ? `${age} ago` : null;
    }
  }
}

export function taskRowID(taskID: string) {
  return `task-row-${taskID}`;
}

export function chatHref(sessionID: string) {
  return `/copilot?sessionId=${encodeURIComponent(sessionID)}`;
}

/** The chat's query key for the task its sidebar opens on arrival. */
export const TASK_PARAM = "task";

/** Home, on its Tasks tab: the board of every task. */
export const BOARD_HREF = "/home?show=tasks";

type Placed = Pick<Task, "id" | "session_ids" | "work_session_id">;

/** Where a task opens: the chat it was asked for in, with the task open in
 *  that chat's sidebar. The chat doing the work stands in when it is the
 *  only one; a task no chat holds has nowhere to open. */
export function taskHref(task: Placed) {
  const asked = task.session_ids?.find((id) => id !== task.work_session_id);
  const chat = asked ?? task.work_session_id ?? null;
  if (!chat) return null;
  return `${chatHref(chat)}&${TASK_PARAM}=${encodeURIComponent(task.id)}`;
}
