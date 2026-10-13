import { z } from "zod/v4";
import { Schedule, scheduleTaskSchema } from "./catalog-schedule";
import { asNumber } from "./calculations";

export type ScheduleProps = z.infer<typeof Schedule.props>;
export type ScheduleTask = z.infer<typeof scheduleTaskSchema>;
const draftSchema = scheduleTaskSchema.extend({
  minutes: z.union([z.number(), z.string().max(10)]),
});
export type ScheduleDraft = z.infer<typeof draftSchema>;

export function readScheduleTasks(
  value: unknown,
  defaults: ScheduleTask[],
): ScheduleDraft[] {
  const parsed = z.array(draftSchema).max(12).safeParse(value);
  if (
    !parsed.success ||
    parsed.data.length !== defaults.length ||
    new Set(parsed.data.map((task) => task.id)).size !== defaults.length ||
    parsed.data.some((task) => !defaults.some((item) => item.id === task.id))
  )
    return defaults;
  return parsed.data.map((task) => ({
    ...defaults.find((item) => item.id === task.id)!,
    minutes: task.minutes,
    done: task.done,
  }));
}

export function durationError(value: unknown) {
  const minutes = asNumber(value);
  return minutes !== null &&
    Number.isInteger(minutes) &&
    minutes > 0 &&
    minutes <= 20_160
    ? ""
    : "Enter a whole number of minutes from 1 to 20160.";
}

export function clockMinutes(value: string) {
  const [hours, minutes] = value.split(":").map(Number);
  return hours * 60 + minutes;
}

export function scheduleTime(value: number, days: ScheduleProps["days"]) {
  const day = Math.floor(value / 1440);
  const time = value % 1440;
  return `${days[day].label} ${String(Math.floor(time / 60)).padStart(2, "0")}:${String(time % 60).padStart(2, "0")}`;
}

export function calculateSchedule(
  days: ScheduleProps["days"],
  tasks: ScheduleDraft[],
) {
  const windows = days.flatMap((day, index) =>
    day.windows.map((window) => ({
      start: index * 1440 + clockMinutes(window.start),
      end: index * 1440 + clockMinutes(window.end),
    })),
  );
  const capacity = windows.reduce(
    (sum, window) => sum + window.end - window.start,
    0,
  );
  if (tasks.some((task) => !task.done && durationError(task.minutes)))
    return null;
  let cursor = windows[0].start;
  let required = 0;
  let unscheduled = 0;
  let blocked = false;
  const allocations = tasks.map((task) => {
    const slots: { start: number; end: number }[] = [];
    if (task.done) return { task, slots, remaining: 0, conflict: "" };
    let remaining = asNumber(task.minutes)!;
    required += remaining;
    if (task.earliest)
      cursor = Math.max(
        cursor,
        task.earliest.day * 1440 + clockMinutes(task.earliest.time),
      );
    const deadline = task.deadline
      ? task.deadline.day * 1440 + clockMinutes(task.deadline.time)
      : Infinity;
    if (!blocked) {
      for (const window of windows) {
        const start = Math.max(cursor, window.start);
        const end = Math.min(window.end, deadline);
        if (start >= end || (!task.splittable && end - start < remaining))
          continue;
        const used = Math.min(remaining, end - start);
        slots.push({ start, end: start + used });
        cursor = start + used;
        remaining -= used;
        if (!remaining) break;
      }
    }
    const conflict = remaining
      ? blocked
        ? "Waiting for an earlier task"
        : task.deadline
          ? "Cannot finish before its deadline"
          : "Does not fit the available windows"
      : "";
    unscheduled += remaining;
    blocked ||= remaining > 0;
    return { task, slots, remaining, conflict };
  });
  const allocated = required - unscheduled;
  return {
    capacity,
    required,
    unscheduled,
    buffer: capacity - required,
    unused: capacity - allocated,
    feasible: unscheduled === 0,
    allocations,
  };
}
