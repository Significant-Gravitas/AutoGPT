import { z } from "zod/v4";
import { asNumber } from "./calculations";

const clockTime = z.string().regex(/^(?:[01]\d|2[0-3]):[0-5]\d$/);
const boundary = z.object({
  day: z.number().int().min(0).max(13),
  time: clockTime,
});

export const scheduleTaskSchema = z.object({
  id: z.string().min(1).max(80),
  title: z.string().min(1).max(200),
  minutes: z.number().int().min(1).max(20_160),
  done: z.boolean(),
  splittable: z.boolean(),
  earliest: boundary.optional(),
  deadline: boundary.optional(),
});

export const scheduleSchema = z
  .object({
    name: z.string().min(1).max(200),
    title: z.string().min(1).max(200),
    days: z
      .array(
        z.object({
          label: z.string().min(1).max(80),
          windows: z
            .array(z.object({ start: clockTime, end: clockTime }))
            .min(1)
            .max(4),
        }),
      )
      .min(1)
      .max(14),
    tasks: z.array(scheduleTaskSchema).min(1).max(12),
  })
  .superRefine(({ days, tasks }, context) => {
    for (const message of scheduleErrors(days, tasks))
      context.addIssue({ code: "custom", message });
  });

type ScheduleProps = z.infer<typeof scheduleSchema>;
export type ScheduleDraft = Omit<
  z.infer<typeof scheduleTaskSchema>,
  "minutes"
> & {
  minutes: number | string;
};

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

function scheduleErrors(
  days: ScheduleProps["days"],
  tasks: ScheduleProps["tasks"],
) {
  const errors: string[] = [];
  if (new Set(days.map((day) => day.label)).size !== days.length)
    errors.push("Use unique day labels.");
  if (new Set(tasks.map((task) => task.id)).size !== tasks.length)
    errors.push("Use unique task ids.");
  for (const day of days) {
    let end = "00:00";
    for (const window of day.windows) {
      if (window.start < end || window.start >= window.end)
        errors.push(
          "Availability windows must be ordered, nonoverlapping and start before they end.",
        );
      end = window.end;
    }
  }
  for (const task of tasks) {
    for (const value of [task.earliest, task.deadline])
      if (value && value.day >= days.length)
        errors.push(
          "Task time constraints must reference an existing zero-based day index.",
        );
    if (
      task.earliest &&
      task.deadline &&
      (task.earliest.day > task.deadline.day ||
        (task.earliest.day === task.deadline.day &&
          task.earliest.time >= task.deadline.time))
    )
      errors.push("A task must start before its deadline.");
  }
  return errors;
}
