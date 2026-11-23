import type { Task } from "@/app/api/__generated__/models/task";
import { TaskStatus } from "@/app/api/__generated__/models/taskStatus";
import { startTimeOf } from "./schedule";

const MINUTE = 60_000;
/** How soon a start counts as near enough to watch for. */
const SOON_MS = 10 * MINUTE;
/** How long past its start a to-do is still expected to start by itself. */
const LATE_MS = 15 * MINUTE;

/** Whether a task can change without the user: it is being worked or
 *  waits on an answer or a free slot, an event is queued to start it, or
 *  its start time
 *  (a scheduled, repeating or watching task's next run) is near. */
export function canChangeByItself(task: Task, now: number) {
  if (task.is_queued) return true;
  if (
    task.status === TaskStatus.IN_PROGRESS ||
    task.status === TaskStatus.NEEDS_INPUT
  ) {
    return true;
  }
  if (task.status !== TaskStatus.TODO) return false;
  if (task.missed_start || task.repeat?.paused) return false;
  if ((task.events_waiting ?? 0) > 0) return true;
  const startsIn = startTimeOf(task.start_at) - now;
  return startsIn <= SOON_MS && startsIn >= -LATE_MS;
}
