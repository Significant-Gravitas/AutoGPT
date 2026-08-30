import { DelegatedTask } from "@/app/api/__generated__/models/delegatedTask";
import { DelegatedTaskStatus } from "@/app/api/__generated__/models/delegatedTaskStatus";
import {
  Alert02Icon,
  CancelCircleIcon,
  CheckmarkCircle02Icon,
  Clock01Icon,
  DashboardSquare01Icon,
  Progress02Icon,
  RemoveCircleIcon,
} from "@hugeicons/core-free-icons";
import type { OrbVariant } from "@/components/atoms/Orb/helpers";
import type { IconSvgElement } from "@hugeicons/react";

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

/** Status glyphs for the table cells. WORKING has none — it animates an Orb
 *  instead, so the row shows the work is still moving. */
const STATUS_ICONS: Record<DelegatedTaskStatus, IconSvgElement | null> = {
  QUEUED: Clock01Icon,
  WORKING: null,
  WAITING_USER: Alert02Icon,
  DONE: CheckmarkCircle02Icon,
  FAILED: CancelCircleIcon,
  CANCELLED: RemoveCircleIcon,
};

/** Only the glyph carries the status color; the label stays neutral so a
 *  column of chips doesn't read as a wall of colored text. */
const STATUS_ICON_CLASSES: Record<DelegatedTaskStatus, string> = {
  QUEUED: "text-zinc-400",
  WORKING: "text-zinc-400",
  WAITING_USER: "text-amber-500",
  DONE: "text-emerald-500",
  FAILED: "text-red-500",
  CANCELLED: "text-zinc-400",
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

export const TASK_FILTER_ALL_ICON = DashboardSquare01Icon;

/** Filter-chip buckets for task tables; icon colors mirror the badge
 *  variants the status cells render with. */
export const TASK_TABLE_FILTERS = [
  { key: "active", label: "Active", icon: Progress02Icon, dot: "#3b82f6" },
  { key: "waiting", label: "Needs you", icon: Alert02Icon, dot: "#f59e0b" },
  {
    key: "done",
    label: "Done",
    icon: CheckmarkCircle02Icon,
    dot: "#10b981",
  },
  { key: "failed", label: "Failed", icon: CancelCircleIcon, dot: "#ef4444" },
  {
    key: "cancelled",
    label: "Cancelled",
    icon: RemoveCircleIcon,
    dot: "#a1a1aa",
  },
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

export function getStatusIcon(
  status: DelegatedTaskStatus,
): IconSvgElement | null {
  return STATUS_ICONS[status];
}

export function getStatusIconClass(status: DelegatedTaskStatus): string {
  return STATUS_ICON_CLASSES[status];
}

const ORB_VARIANTS: OrbVariant[] = ["S1", "S2", "S3", "S4", "S5"];

/** Each task keeps its own sweep, derived from the id so it stays the same
 *  across refetches — a board of working rows pulses out of step rather than
 *  in unison. */
export function getTaskOrbVariant(taskId: string): OrbVariant {
  let hash = 0;
  for (const char of taskId) hash = (hash * 31 + char.charCodeAt(0)) % 100_000;
  return ORB_VARIANTS[hash % ORB_VARIANTS.length];
}

/** Credits are stored as hundredths of a dollar, matching the expert budget
 *  meter, so a sub-cent task reads "<$0.01" rather than "$0.00". */
export function formatSpend(credits: number): string {
  if (credits <= 0) return "$0.00";
  const dollars = credits / 100;
  return dollars < 0.01 ? "<$0.01" : `$${dollars.toFixed(2)}`;
}

/** How long an open task has been running, or how long ago a closed one
 *  finished. A closed task's *duration* is the wrong reading here: a run that
 *  took 30 seconds this morning would sit at "just now" all day. Coarse on
 *  purpose — a live seconds counter would re-render the list every second for
 *  no decision-making value. */
export function formatElapsed(
  task: DelegatedTask,
  now: number = Date.now(),
): string {
  // Orval types these as Date, but a hydrated cache or an MSW fixture can
  // still hand back the raw ISO string — coerce rather than trust the type.
  const start = toMillis(task.created_at);
  if (start === null) return "";
  if (isOpenTask(task)) return formatGap(now - start) ?? "just now";

  const finished = toMillis(task.updated_at) ?? start;
  const gap = formatGap(now - finished);
  return gap === null ? "just now" : `${gap} ago`;
}

/** Coarse "4m" / "3h" / "2d", or null under a minute. Negative gaps (a server
 *  clock running ahead) read as null rather than a negative count. */
function formatGap(millis: number): string | null {
  const minutes = Math.floor(Math.max(millis, 0) / 60_000);
  if (minutes < 1) return null;
  if (minutes < 60) return `${minutes}m`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h`;
  return `${Math.floor(hours / 24)}d`;
}

function toMillis(value: Date | string): number | null {
  const millis = value instanceof Date ? value.getTime() : Date.parse(value);
  return Number.isNaN(millis) ? null : millis;
}
