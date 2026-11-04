import type { Tone } from "@/components/styles/tones";
import { statusOf } from "../../data/helpers";
import type { InboxState, Task } from "../../data/types";

export const STATE_TONE: Record<InboxState, Tone> = {
  running: "info",
  waiting: "waiting",
  done: "ok",
  failed: "error",
};

/** The task's one status word, from the shared vocabulary. */
export function statusLabel(task: Task) {
  return statusOf(task).word;
}

/** The colour that goes with `statusLabel`. */
export function statusTone(task: Task): Tone {
  return statusOf(task).tone;
}

/** The quiet button beside a card's main one, which opens the panel. */
export function openLabel(task: Task, hasAction: boolean) {
  if (!task.ask) return "Details";
  if (!hasAction) return "Answer";
  return (task.ask.options?.length ?? 0) > 1 ? "Other answers" : "Details";
}
