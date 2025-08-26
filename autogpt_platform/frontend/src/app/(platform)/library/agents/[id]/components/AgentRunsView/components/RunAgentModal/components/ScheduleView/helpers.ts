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
