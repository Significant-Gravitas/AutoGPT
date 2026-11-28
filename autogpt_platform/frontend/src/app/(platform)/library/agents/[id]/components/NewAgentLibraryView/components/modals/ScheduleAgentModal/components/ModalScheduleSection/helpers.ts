import { z } from "zod";

export const timeRegex = /^([01]?\d|2[0-3]):([0-5]\d)$/;
// The backend's limit, which counts code points as Array.from does.
export const MAX_SCHEDULE_NAME_LENGTH = 200;

export const scheduleFormSchema = z.object({
  scheduleName: z
    .string()
    .trim()
    .min(1, "Schedule name is required")
    .refine(
      (name) => Array.from(name).length <= MAX_SCHEDULE_NAME_LENGTH,
      `Shorten the schedule name to ${MAX_SCHEDULE_NAME_LENGTH} characters or fewer`,
    ),
  time: z.string().trim().regex(timeRegex, "Use HH:MM (24h)").optional(),
});

export type ScheduleFormValues = z.infer<typeof scheduleFormSchema>;

export function validateSchedule(
  values: Partial<ScheduleFormValues>,
): Partial<Record<keyof ScheduleFormValues, string>> {
  const result = scheduleFormSchema.safeParse({
    scheduleName: values.scheduleName ?? "",
    ...(values.time !== undefined ? { time: values.time } : {}),
  });

  if (result.success) return {};

  const fieldErrors: Partial<Record<keyof ScheduleFormValues, string>> = {};
  for (const issue of result.error.issues) {
    const path = issue.path[0] as keyof ScheduleFormValues | undefined;
    if (path && !fieldErrors[path]) fieldErrors[path] = issue.message;
  }
  return fieldErrors;
}

export function defaultScheduleNameFor(workflowName: string): string {
  const name = Array.from(`Run ${workflowName}`);
  if (name.length <= MAX_SCHEDULE_NAME_LENGTH) return name.join("");
  const kept = name
    .slice(0, MAX_SCHEDULE_NAME_LENGTH - 1)
    .join("")
    .trimEnd();
  return `${kept}…`;
}
