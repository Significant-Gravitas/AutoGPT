import type { Tone } from "@/components/styles/tones";
import type { InboxState, Task } from "../../data/types";

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

/** The quiet button beside a card's main one, which opens the panel. */
export function openLabel(task: Task, hasAction: boolean) {
  if (!task.ask) return "Details";
  if (!hasAction) return "Answer";
  return (task.ask.options?.length ?? 0) > 1 ? "Other answers" : "Details";
}
