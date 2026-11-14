import type { SessionSummaryResponse } from "@/app/api/__generated__/models/sessionSummaryResponse";
import type { Task } from "@/app/api/__generated__/models/task";
import { TaskStatus } from "@/app/api/__generated__/models/taskStatus";
import { laneOf } from "@/app/(platform)/tasks/helpers";

const DAY = 24 * 60 * 60 * 1000;

function at(value: Date | string | null | undefined) {
  return value ? new Date(value).getTime() : 0;
}

function dayStart(ms: number) {
  const date = new Date(ms);
  date.setHours(0, 0, 0, 0);
  return date.getTime();
}

export function span(seconds: number) {
  if (seconds <= 0) return "0m";
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) return `${Math.max(1, minutes)}m`;
  const rest = minutes % 60;
  return rest ? `${Math.floor(minutes / 60)}h ${rest}m` : `${minutes / 60}h`;
}

export interface DayStat {
  label: string;
  isToday: boolean;
  done: Task[];
  started: number;
  chats: number;
}

export interface OwnerStat {
  /** Null is Otto. */
  ownerID: string | null;
  done: number;
  working: number;
  needs: number;
}

/** The last seven days of work, counted from the tasks and chats the page
 *  already has: what finished each day, how long it ran, and who did it. */
export function buildWeekStats(
  tasks: Task[],
  sessions: SessionSummaryResponse[],
  now: number,
) {
  const own = tasks.filter((task) => !task.parent_task_id && !task.is_archived);
  const today = dayStart(now);
  const weekStart = today - 6 * DAY;
  const finishedAt = (task: Task) => at(task.completed_at ?? task.updated_at);
  const doneWeek = own.filter(
    (task) => task.status === TaskStatus.DONE && finishedAt(task) >= weekStart,
  );

  const days: DayStat[] = Array.from({ length: 7 }, (_, index) => {
    const start = weekStart + index * DAY;
    const isIn = (ms: number) => ms >= start && ms < start + DAY;
    return {
      label: new Date(start).toLocaleDateString(undefined, {
        weekday: "short",
      }),
      isToday: index === 6,
      done: doneWeek.filter((task) => isIn(finishedAt(task))),
      started: own.filter((task) => isIn(at(task.created_at))).length,
      chats: sessions.filter((chat) => isIn(at(chat.updated_at))).length,
    };
  });

  const runs = doneWeek
    .filter((task) => task.started_at && task.completed_at)
    .map((task) => (at(task.completed_at) - at(task.started_at)) / 1000);
  const workSeconds = runs.reduce((sum, seconds) => sum + seconds, 0);

  const owners = new Map<string | null, OwnerStat>();
  for (const task of own) {
    const ownerID = task.owner_expert_id ?? null;
    const stat = owners.get(ownerID) ?? {
      ownerID,
      done: 0,
      working: 0,
      needs: 0,
    };
    owners.set(ownerID, {
      ...stat,
      done: stat.done + (doneWeek.includes(task) ? 1 : 0),
      working: stat.working + (task.status === TaskStatus.IN_PROGRESS ? 1 : 0),
      needs: stat.needs + (laneOf(task) === "needs_you" ? 1 : 0),
    });
  }

  const settled = own.filter(
    (task) =>
      (task.status === TaskStatus.DONE || task.status === TaskStatus.FAILED) &&
      finishedAt(task) >= weekStart,
  );

  return {
    days,
    doneWeek: doneWeek.length,
    startedWeek: days.reduce((sum, day) => sum + day.started, 0),
    chatsWeek: days.reduce((sum, day) => sum + day.chats, 0),
    workSeconds,
    avgSeconds: runs.length ? workSeconds / runs.length : 0,
    /** Of the tasks that ended this week, the share that finished. */
    doneRate: settled.length
      ? Math.round((doneWeek.length / settled.length) * 100)
      : null,
    owners: [...owners.values()],
  };
}
