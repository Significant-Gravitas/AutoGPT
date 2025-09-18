import { z } from "zod";

export const timeRegex = /^([01]?\d|2[0-3]):([0-5]\d)$/;

export const scheduleFormSchema = z.object({
  scheduleName: z.string().trim().min(1, "Schedule name is required"),
  time: z.string().trim().regex(timeRegex, "Use HH:MM (24h)"),
});

export type ScheduleFormValues = z.infer<typeof scheduleFormSchema>;

export function validateSchedule(
  values: Partial<ScheduleFormValues>,
): Partial<Record<keyof ScheduleFormValues, string>> {
  const result = scheduleFormSchema.safeParse({
    scheduleName: values.scheduleName ?? "",
    time: values.time ?? "",
  });

  if (result.success) return {};

  const fieldErrors: Partial<Record<keyof ScheduleFormValues, string>> = {};
  for (const issue of result.error.issues) {
    const path = issue.path[0] as keyof ScheduleFormValues | undefined;
    if (path && !fieldErrors[path]) fieldErrors[path] = issue.message;
  }
  return fieldErrors;
}

export type ParsedCron = {
  repeat: "daily" | "weekly";
  selectedDays: string[]; // for weekly, 0-6 (0=Sun) as strings
  time: string; // HH:MM
};

export function parseCronToForm(cron: string): ParsedCron | undefined {
  if (!cron) return undefined;
  const parts = cron.trim().split(/\s+/);
  if (parts.length !== 5) return undefined;
  const [minute, hour, _dom, _mon, dow] = parts;
  const hh = String(hour ?? "0").padStart(2, "0");
  const mm = String(minute ?? "0").padStart(2, "0");
  const time = `${hh}:${mm}`; // Cron is stored in UTC; we keep raw HH:MM

  if (dow && dow !== "*") {
    return { repeat: "weekly", selectedDays: dow.split(","), time };
  }

  return { repeat: "daily", selectedDays: [], time };
}
