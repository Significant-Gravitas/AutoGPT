import type { Task } from "@/app/api/__generated__/models/task";
import { TaskStatus } from "@/app/api/__generated__/models/taskStatus";
import { laneOf, statusAge } from "@/app/(platform)/tasks/helpers";

/** How far back a finished task still counts as news. */
const NEWS_MS = 72 * 60 * 60 * 1000;

function timeOf(value: Date | string | null | undefined) {
  return value ? new Date(value).getTime() : 0;
}

/** The day's tasks, sorted into the three things a person checks first:
 *  what waits on them, what is ready to read and what is being worked on. */
export function sortToday(tasks: Task[], now: number) {
  const own = tasks.filter((task) => !task.parent_task_id && !task.is_archived);
  const finishedAt = (task: Task) =>
    timeOf(task.completed_at ?? task.updated_at);
  return {
    needs: own
      .filter((task) => laneOf(task) === "needs_you")
      .toSorted((a, b) => timeOf(b.updated_at) - timeOf(a.updated_at)),
    ready: own
      .filter(
        (task) =>
          task.status === TaskStatus.DONE && now - finishedAt(task) < NEWS_MS,
      )
      .toSorted((a, b) => finishedAt(b) - finishedAt(a)),
    working: own
      .filter((task) => task.status === TaskStatus.IN_PROGRESS)
      .toSorted((a, b) => timeOf(b.updated_at) - timeOf(a.updated_at)),
  };
}

/** What sits at a task row's end: how long it has waited, run or been done. */
export function metaOf(task: Task, now: number) {
  if (task.status === TaskStatus.FAILED) return "Didn't finish";
  return statusAge(task, now) ?? "";
}

export function dayPart(now: number) {
  const hour = new Date(now).getHours();
  if (hour < 12) return "This morning";
  return hour < 18 ? "This afternoon" : "This evening";
}

/** What the brief is called at this hour. */
export function briefName(now: number) {
  return `${dayPart(now)
    .replace("This ", "")
    .replace(/^./, (c) => c.toUpperCase())} brief`;
}
