import type { Task } from "@/app/api/__generated__/models/task";
import { TaskStatus } from "@/app/api/__generated__/models/taskStatus";

/** What a waiting surface needs of a task's rhythm: its words, whether it
 *  is paused and, for a task started by an event, what it watches. */
export interface RepeatLike {
  label: string;
  paused?: boolean;
  watch?: WatchLike | null;
}

export interface WatchLike {
  line: string;
  url?: string | null;
}

type Timed = Pick<Task, "status" | "start_at" | "missed_start"> & {
  repeat?: RepeatLike | null;
};
type When = Date | string | number | null | undefined;

/** A to-do task waiting for its start time, one whose time passed while
 *  nothing could start it, a repeating one waiting for its next run, one
 *  waiting for the event it watches (or either of those paused). */
export type ScheduleState =
  | "scheduled"
  | "missed"
  | "repeating"
  | "watching"
  | "paused";

/** The one word a pill or status row shows for each. */
export const SCHEDULE_WORDS: Record<ScheduleState, string> = {
  scheduled: "Scheduled",
  missed: "Missed start",
  repeating: "Repeats",
  watching: "Watching",
  paused: "Paused",
};

const MINUTE = 60_000;
const DAY = 24 * 60 * MINUTE;
const WEEK_DAYS = 7;

/** A task that runs again and again, whatever it is doing now. */
export function isRepeating(task: Timed) {
  return task.repeat != null;
}

/** A task that runs each time an event happens, whatever it is doing now. */
export function isWatching(task: Timed) {
  return task.repeat?.watch != null;
}

/** Between runs of a repeating or watching task, paused or not. */
export function isBetweenRuns(state: ScheduleState | null) {
  return state === "repeating" || state === "watching" || state === "paused";
}

/** Waiting for its start time, or a repeating task waiting (or paused)
 *  between runs. */
export function isScheduled(task: Timed) {
  return (
    task.status === TaskStatus.TODO &&
    (task.start_at != null || isRepeating(task))
  );
}

export function isMissed(task: Timed) {
  return task.status === TaskStatus.TODO && task.missed_start === true;
}

export function scheduleStateOf(task: Timed): ScheduleState | null {
  if (isMissed(task)) return "missed";
  if (!isScheduled(task)) return null;
  if (task.repeat?.paused) return "paused";
  if (task.repeat?.watch) return "watching";
  if (task.repeat) return "repeating";
  return "scheduled";
}

/** A countdown label ("Starts in 4 min", "Starts tomorrow 9:00 AM") said as
 *  the next run of a repeating task ("next run in 4 min"). */
export function asNextRun(startsLabel: string) {
  if (startsLabel === "Starting now") return "next run starting now";
  return startsLabel.replace(/^Starts/, "next run");
}

/** A repeating task's rhythm and where it is: "Every day at 08:00 · next
 *  run tomorrow 8:00 AM", or "… · Paused". */
export function rhythmLine(label: string, isPaused: boolean, starts?: string) {
  if (isPaused) return `${label} · Paused`;
  return starts ? `${label} · ${asNextRun(starts)}` : label;
}

/** Where a repeating or watching task stands between runs: its rhythm and
 *  next run, or what it watches ("Watching · PR opened on acme/web"), or
 *  "… · Paused". */
export function waitingLine(repeat: RepeatLike, starts?: string) {
  if (!repeat.watch) return rhythmLine(repeat.label, !!repeat.paused, starts);
  if (repeat.paused) return `${repeat.watch.line} · Paused`;
  return `Watching · ${repeat.watch.line}`;
}

/** Epoch milliseconds of a start time; NaN when there is none. */
export function startTimeOf(startAt: When) {
  if (startAt == null || startAt === "") return Number.NaN;
  return new Date(startAt).getTime();
}

function startOfDay(at: number) {
  const day = new Date(at);
  day.setHours(0, 0, 0, 0);
  return day.getTime();
}

/** Whole calendar days from `now` to `at`: 0 today, 1 tomorrow. */
function daysAway(at: number, now: number) {
  return Math.round((startOfDay(at) - startOfDay(now)) / DAY);
}

function clockTime(at: number) {
  return new Date(at).toLocaleTimeString(undefined, {
    hour: "numeric",
    minute: "2-digit",
  });
}

/** The exact start in the user's local time: "10:05 AM", "Tomorrow, 9:00
 *  AM", "Mon, 9:00 AM", or a date further out. */
export function formatStartTime(startAt: When, now: number) {
  const at = startTimeOf(startAt);
  if (!Number.isFinite(at)) return "";
  const days = daysAway(at, now);
  if (days === 0) return clockTime(at);
  if (days === 1) return `Tomorrow, ${clockTime(at)}`;
  const date = new Date(at).toLocaleDateString(
    undefined,
    days > 0 && days < WEEK_DAYS
      ? { weekday: "short" }
      : { day: "numeric", month: "short" },
  );
  return `${date}, ${clockTime(at)}`;
}

/** How soon it starts: "Starts in 40s", "Starts in 4 min", "Starts at 10:05
 *  AM", "Starts tomorrow 9:00 AM", "Starts Mon 9:00 AM". */
export function formatStartsIn(startAt: When, now: number) {
  const at = startTimeOf(startAt);
  if (!Number.isFinite(at)) return "";
  const left = at - now;
  if (left <= 0) return "Starting now";
  if (left < MINUTE) return `Starts in ${Math.ceil(left / 1000)}s`;
  if (left < 60 * MINUTE) return `Starts in ${Math.ceil(left / MINUTE)} min`;
  const days = daysAway(at, now);
  if (days === 0) return `Starts at ${clockTime(at)}`;
  if (days === 1) return `Starts tomorrow ${clockTime(at)}`;
  return `Starts ${formatStartTime(at, now).replace(",", "")}`;
}

/** A scheduled task due before today ends. */
export function startsToday(task: Timed, now: number) {
  if (!isScheduled(task)) return false;
  return daysAway(startTimeOf(task.start_at), now) <= 0;
}

/** Soonest start first; tasks without one go last. */
export function bySoonestStart(a: Timed, b: Timed) {
  const left = startTimeOf(a.start_at);
  const right = startTimeOf(b.start_at);
  return (
    (Number.isFinite(left) ? left : Infinity) -
    (Number.isFinite(right) ? right : Infinity)
  );
}

export interface QuickStart {
  label: string;
  /** The start time this choice means, counted from `now`. */
  at: (now: number) => Date;
}

function tomorrowAtNine(now: number) {
  const day = new Date(now);
  day.setDate(day.getDate() + 1);
  day.setHours(9, 0, 0, 0);
  return day;
}

/** The few start times offered with one click. */
export const QUICK_STARTS: QuickStart[] = [
  { label: "In 5 min", at: (now) => new Date(now + 5 * MINUTE) },
  { label: "In 1 hour", at: (now) => new Date(now + 60 * MINUTE) },
  { label: "Tomorrow 9:00", at: tomorrowAtNine },
];

/** "YYYY-MM-DDTHH:MM" in local time, the value DateTimeInput reads. */
export function toLocalInputValue(startAt: When) {
  const at = startTimeOf(startAt);
  if (!Number.isFinite(at)) return "";
  const d = new Date(at);
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

/** The local "YYYY-MM-DDTHH:MM" value back as a time; null if unreadable. */
export function fromLocalInputValue(value: string | undefined) {
  if (!value) return null;
  const at = new Date(value.replace(" ", "T"));
  return Number.isNaN(at.getTime()) ? null : at;
}
