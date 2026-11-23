import type { Task } from "@/app/api/__generated__/models/task";
import { TaskStatus } from "@/app/api/__generated__/models/taskStatus";
import { onlineManager } from "@tanstack/react-query";

const NOT_FOUND = 404;
const CONFLICT = 409;
const ALREADY = /already/i;

/** The HTTP status a request failed with; null when no answer came. */
export function httpStatusOf(error: unknown) {
  return error instanceof Error &&
    "status" in error &&
    typeof error.status === "number"
    ? error.status
    : null;
}

/** The request never reached the server: offline, or the network failed. */
export function isUnreachable(error: unknown) {
  if (httpStatusOf(error) !== null) return false;
  return error instanceof TypeError || !onlineManager.isOnline();
}

/** The task is gone: archived or removed, in this tab or another. */
export function isTaskGone(error: unknown) {
  return httpStatusOf(error) === NOT_FOUND;
}

/** A start refused because the task is already being worked on: what the
 *  user asked for is true, so it is not an error. A 409 for anything else
 *  (the chat busy with another message) still is. */
export function isAlreadyRunning(error: unknown, before: Task | null) {
  if (httpStatusOf(error) !== CONFLICT) return false;
  const wasRunning =
    before?.status === TaskStatus.IN_PROGRESS ||
    before?.status === TaskStatus.NEEDS_INPUT;
  return wasRunning || (error instanceof Error && ALREADY.test(error.message));
}
