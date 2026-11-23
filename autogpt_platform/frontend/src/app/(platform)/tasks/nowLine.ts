import type { Task } from "@/app/api/__generated__/models/task";
import { TaskStatus } from "@/app/api/__generated__/models/taskStatus";
import { isQueued, QUEUED_LINE } from "./queue";

/** Leading list, heading and quote marks, and inline emphasis. */
const MARKDOWN_LEAD = /^(?:#{1,6}\s+|>\s*|[-*+]\s+|\d+[.)]\s+)+/;
const MARKDOWN_INLINE = /(\*\*|__|`)/g;

/** The first line of text that says something, without its markdown. */
export function firstLine(text: string | null | undefined) {
  const line = text
    ?.split("\n")
    .map((entry) => entry.replace(MARKDOWN_LEAD, "").trim())
    .find((entry) => entry.length > 0);
  return line ? line.replace(MARKDOWN_INLINE, "").trim() || null : null;
}

/** What is happening now, in one line: the worker's current step while it
 *  works (or that it waits for a slot), the result once done, the reason
 *  when it failed. */
export function nowLineOf(task: Task) {
  if (isQueued(task)) return QUEUED_LINE;
  switch (task.status) {
    case TaskStatus.IN_PROGRESS:
    case TaskStatus.NEEDS_INPUT:
      return firstLine(task.status_text);
    case TaskStatus.DONE:
      return firstLine(task.result);
    case TaskStatus.FAILED:
      return firstLine(task.failure_reason) ?? firstLine(task.result);
    default:
      return null;
  }
}

/** "3 / 5" when the task is split into subtasks. */
export function subtaskProgressOf(task: Task) {
  const total = task.subtask_count ?? 0;
  if (total <= 0) return null;
  return { done: Math.min(task.subtasks_done ?? 0, total), total };
}
