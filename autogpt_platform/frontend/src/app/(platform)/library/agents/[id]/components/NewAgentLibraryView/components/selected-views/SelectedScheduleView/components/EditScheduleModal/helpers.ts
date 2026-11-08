import type { GraphExecutionJobInfo } from "@/app/api/__generated__/models/graphExecutionJobInfo";

export const REPEAT_OPTIONS = [
  { value: "daily", label: "Daily" },
  { value: "weekly", label: "Weekly" },
];

export const DAY_ITEMS = [
  { value: "0", label: "Su" },
  { value: "1", label: "Mo" },
  { value: "2", label: "Tu" },
  { value: "3", label: "We" },
  { value: "4", label: "Th" },
  { value: "5", label: "Fr" },
  { value: "6", label: "Sa" },
];

export interface EditScheduleForm {
  name: string;
  // False when the cron is something this form can't express (hourly,
  // monthly, ranges, steps...). Saving then sends the name only, so the
  // schedule's timing is never rewritten into a daily or weekly one.
  canEditTiming: boolean;
  repeat: string;
  selectedDays: string[];
  time: string;
}

export function formFromSchedule(
  schedule: GraphExecutionJobInfo,
): EditScheduleForm {
  const timing = parseEditableCron(schedule.cron);
  return {
    name: schedule.name,
    canEditTiming: timing !== undefined,
    repeat: timing?.repeat ?? "daily",
    selectedDays: timing?.selectedDays ?? [],
    time: timing?.time ?? "00:00",
  };
}

export function buildCron(form: EditScheduleForm): string {
  const [hh, mm] = form.time.split(":");
  const minute = Number(mm || 0);
  const hour = Number(hh || 0);
  const days =
    form.repeat === "weekly"
      ? [...form.selectedDays].sort((a, b) => Number(a) - Number(b)).join(",")
      : "*";
  return `${minute} ${hour} * * ${days}`;
}

function parseEditableCron(cron: string) {
  const parts = cron.trim().split(/\s+/);
  if (parts.length !== 5) return undefined;
  const [minute, hour, dayOfMonth, month, dayOfWeek] = parts;
  if (!/^\d{1,2}$/.test(minute) || !/^\d{1,2}$/.test(hour)) return undefined;
  if (dayOfMonth !== "*" || month !== "*") return undefined;
  if (Number(minute) > 59 || Number(hour) > 23) return undefined;

  const time = `${hour.padStart(2, "0")}:${minute.padStart(2, "0")}`;
  if (dayOfWeek === "*") return { repeat: "daily", selectedDays: [], time };

  const days = dayOfWeek.split(",");
  if (!days.every((day) => /^[0-6]$/.test(day))) return undefined;
  return { repeat: "weekly", selectedDays: days, time };
}
