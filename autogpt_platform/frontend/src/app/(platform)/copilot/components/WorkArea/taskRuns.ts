import type { Task } from "@/app/api/__generated__/models/task";
import type { TaskRun } from "@/app/api/__generated__/models/taskRun";
import type { Tone } from "@/components/styles/tones";
import {
  Alert02Icon,
  Loading03Icon,
  StopIcon,
  Tick02Icon,
} from "@hugeicons/core-free-icons";
import type { IconSvgElement } from "@hugeicons/react";

export interface RunView {
  word: string;
  icon: IconSvgElement;
  tone: Tone;
  spin?: boolean;
}

const RUN_VIEWS: Record<string, RunView> = {
  done: { word: "Done", icon: Tick02Icon, tone: "ok" },
  failed: { word: "Failed", icon: Alert02Icon, tone: "error" },
  cancelled: { word: "Stopped", icon: StopIcon, tone: "faint" },
};
const RUNNING: RunView = {
  word: "Running",
  icon: Loading03Icon,
  tone: "info",
  spin: true,
};
const UNKNOWN: RunView = { word: "Ended", icon: StopIcon, tone: "faint" };

/** How a run ended; one with no end is running if it is the newest and the
 *  task is being worked on. */
export function runViewOf(run: TaskRun, isNewest: boolean, task: Task) {
  if (run.outcome) return RUN_VIEWS[run.outcome] ?? UNKNOWN;
  const isWorking =
    task.status === "IN_PROGRESS" || task.status === "NEEDS_INPUT";
  return isNewest && isWorking ? RUNNING : UNKNOWN;
}

/** When a run started: "Today, 8:00 AM", "Yesterday, 8:00 AM", "14 Nov,
 *  8:00 AM". */
export function runWhen(startedAt: Date | string, now: number) {
  const at = new Date(startedAt);
  const time = at.toLocaleTimeString(undefined, {
    hour: "numeric",
    minute: "2-digit",
  });
  const today = new Date(now);
  today.setHours(0, 0, 0, 0);
  const day = new Date(at);
  day.setHours(0, 0, 0, 0);
  const daysAgo = Math.round((today.getTime() - day.getTime()) / 86_400_000);
  if (daysAgo === 0) return `Today, ${time}`;
  if (daysAgo === 1) return `Yesterday, ${time}`;
  const date = at.toLocaleDateString(undefined, {
    day: "numeric",
    month: "short",
  });
  return `${date}, ${time}`;
}

/** When a run started and, for one an event started, what happened:
 *  "Today, 8:00 AM · PR #12 opened: Fix login". */
export function runTitle(run: TaskRun, now: number) {
  const when = runWhen(run.started_at, now);
  return run.trigger ? `${when} · ${run.trigger}` : when;
}

/** The Runs group shows for a repeating task, or one run more than once. */
export function showsRuns(task: Task, runs: TaskRun[]) {
  return runs.length > 0 && (task.repeat != null || runs.length > 1);
}
