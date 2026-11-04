import type { Tone } from "@/components/styles/tones";
import type { InboxState, Task, TaskInboxStep } from "../../data/types";

export type RowAction = "answer" | "review" | "retry" | "open";

export function formatDuration(ms: number) {
  const total = Math.max(0, Math.floor(ms / 1000));
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  if (h > 0) return `${h}h ${m}m`;
  if (m > 0) return `${m}m ${s}s`;
  return `${s}s`;
}

export function elapsedFrom(startedAt: string | null, now: number) {
  if (!startedAt) return null;
  const started = new Date(startedAt).getTime();
  if (Number.isNaN(started)) return null;
  return Math.max(0, now - started);
}

export const STATE_LABEL: Record<InboxState, string> = {
  running: "Running",
  waiting: "Waiting",
  done: "Done",
  failed: "Failed",
};

export const STATE_TONE: Record<InboxState, Tone> = {
  running: "info",
  waiting: "waiting",
  done: "ok",
  failed: "error",
};

export function statusLabel(task: Task) {
  if (task.inbox.needs_you) return "Needs you";
  return STATE_LABEL[task.inbox.state];
}

export function secondaryLine(task: Task) {
  if (task.status === "blocked" && task.open_question)
    return task.open_question.question;
  if (task.inbox.last_activity?.text) return task.inbox.last_activity.text;
  if (task.live?.current_step) return task.live.current_step;
  return null;
}

export function rowAction(task: Task): RowAction {
  if (task.inbox.state === "failed") return "retry";
  if (task.status === "blocked" && task.open_question) return "answer";
  if (task.status === "in_review") return "review";
  return "open";
}

export function relativeTime(iso: string, now: number) {
  const then = new Date(iso).getTime();
  if (Number.isNaN(then)) return "";
  const diff = Math.max(0, Math.floor((now - then) / 1000));
  if (diff < 45) return "just now";
  if (diff < 3600) return `${Math.round(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.round(diff / 3600)}h ago`;
  return `${Math.round(diff / 86400)}d ago`;
}

export const STEP_DOT: Record<NonNullable<TaskInboxStep["status"]>, string> = {
  completed: "bg-status-ok",
  running: "bg-info",
  error: "bg-error",
};

export function stepDotClass(step: TaskInboxStep) {
  return step.status ? STEP_DOT[step.status] : "bg-foreground-faint";
}

export function isTypingTarget(target: EventTarget | null) {
  if (!(target instanceof HTMLElement)) return false;
  return (
    target.isContentEditable ||
    target.tagName === "INPUT" ||
    target.tagName === "TEXTAREA" ||
    target.tagName === "SELECT"
  );
}
