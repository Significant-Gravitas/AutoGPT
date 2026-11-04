import { shortAgo } from "../../data/helpers";
import type { Task, TaskState } from "../../data/types";

export const STATE_WORD: Record<TaskState, string> = {
  working: "Running",
  needs_you: "Waiting for you",
  stuck: "Stuck",
  done: "Done",
  stopped: "Cancelled",
};

/** "45s", "4m", "1h 12m", "2d 3h": how long a task has run or took. */
export function formatDuration(ms: number) {
  const s = Math.max(0, Math.floor(ms / 1000));
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m`;
  const h = Math.floor(m / 60);
  if (h < 24) return m % 60 ? `${h}h ${m % 60}m` : `${h}h`;
  return `${Math.floor(h / 24)}d ${h % 24}h`;
}

/** The clock on a card: running time for live tasks, total for finished ones. */
export function cardClock(task: Task, now: number) {
  if (task.state === "stopped") return null;
  if (task.state === "done") {
    return task.elapsed_ms && task.elapsed_ms >= 1000
      ? formatDuration(task.elapsed_ms)
      : null;
  }
  const start = Date.parse(task.started_at ?? task.created_at);
  return Number.isFinite(start) ? formatDuration(now - start) : null;
}

/** The last few steps, oldest first, ready to show. */
export function lastSteps(task: Task, count = 4) {
  return (task.inbox?.steps ?? []).slice(-count).map((step) => ({
    label: step.label,
    time: step.time ? shortAgo(step.time) : null,
  }));
}

/** A waiting card says what it waits on, with our own buttons. */
export function needsAnswer(task: Task) {
  if (task.state !== "needs_you" && task.state !== "stuck") return false;
  return (
    task.approvals.length > 0 ||
    !!task.ask ||
    !!task.blocker ||
    task.raw_status === "awaiting_accept"
  );
}
