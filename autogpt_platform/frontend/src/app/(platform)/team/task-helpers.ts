import { DelegatedTask } from "@/app/api/__generated__/models/delegatedTask";
import { DelegatedTaskStatus } from "@/app/api/__generated__/models/delegatedTaskStatus";

/** A task is "active" while it can still change on its own — mirrors
 *  ``OPEN_TASK_STATUSES`` in the backend's tasks/models.py. */
const OPEN_STATUSES: DelegatedTaskStatus[] = [
  "QUEUED",
  "WORKING",
  "WAITING_USER",
];

const STATUS_LABELS: Record<DelegatedTaskStatus, string> = {
  QUEUED: "Queued",
  WORKING: "Working",
  WAITING_USER: "Needs you",
  DONE: "Done",
  FAILED: "Failed",
  CANCELLED: "Cancelled",
};

type BadgeVariant = "success" | "error" | "warning" | "info";

const STATUS_VARIANTS: Record<DelegatedTaskStatus, BadgeVariant> = {
  QUEUED: "info",
  WORKING: "info",
  WAITING_USER: "warning",
  DONE: "success",
  FAILED: "error",
  CANCELLED: "info",
};

export function isOpenTask(task: DelegatedTask): boolean {
  return OPEN_STATUSES.includes(task.status);
}

/** Filter-chip buckets for task tables; dot colors mirror the badge
 *  variants the status cells render with. */
export const TASK_TABLE_FILTERS = [
  { key: "active", label: "Active", dot: "#3b82f6" },
  { key: "waiting", label: "Needs you", dot: "#f59e0b" },
  { key: "done", label: "Done", dot: "#10b981" },
  { key: "failed", label: "Failed", dot: "#ef4444" },
  { key: "cancelled", label: "Cancelled", dot: "#a1a1aa" },
];

export function getTaskFilterKey(task: DelegatedTask): string {
  if (task.status === "WAITING_USER") return "waiting";
  if (isOpenTask(task)) return "active";
  return task.status.toLowerCase();
}

export function getStatusLabel(status: DelegatedTaskStatus): string {
  return STATUS_LABELS[status];
}

export function getStatusVariant(status: DelegatedTaskStatus): BadgeVariant {
  return STATUS_VARIANTS[status];
}

/** Credits are stored as hundredths of a dollar, matching the expert budget
 *  meter, so a sub-cent task reads "<$0.01" rather than "$0.00". */
export function formatSpend(credits: number): string {
  if (credits <= 0) return "$0.00";
  const dollars = credits / 100;
  return dollars < 0.01 ? "<$0.01" : `$${dollars.toFixed(2)}`;
}

/** How long the task has been open, or how long it took once closed. Coarse
 *  on purpose: a live-ticking seconds counter would re-render the whole list
 *  every second for no decision-making value. */
export function formatElapsed(
  task: DelegatedTask,
  now: number = Date.now(),
): string {
  // Orval types these as Date, but a hydrated cache or an MSW fixture can
  // still hand back the raw ISO string — coerce rather than trust the type.
  const start = toMillis(task.created_at);
  if (start === null) return "";
  const end = isOpenTask(task) ? now : toMillis(task.updated_at);
  const minutes = Math.floor((Math.max(end ?? now, start) - start) / 60_000);
  if (minutes < 1) return "just now";
  if (minutes < 60) return `${minutes}m`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h`;
  return `${Math.floor(hours / 24)}d`;
}

function toMillis(value: Date | string): number | null {
  const millis = value instanceof Date ? value.getTime() : Date.parse(value);
  return Number.isNaN(millis) ? null : millis;
}
