import type { Task } from "@/app/api/__generated__/models/task";
import { TaskStatus } from "@/app/api/__generated__/models/taskStatus";

type Queueable = Pick<Task, "status" | "is_queued">;

/** The pill's word for a started task waiting for a free turn slot. */
export const QUEUED_WORD = "Queued";
/** What a queued task says under its title. */
export const QUEUED_LINE = "Starts when a slot frees";

/** Started, but its run waits for a free turn slot: it is not working yet.
 *  It stays in the Working lane; only its pill and line differ. */
export function isQueued(task: Queueable) {
  return task.status === TaskStatus.IN_PROGRESS && task.is_queued === true;
}
