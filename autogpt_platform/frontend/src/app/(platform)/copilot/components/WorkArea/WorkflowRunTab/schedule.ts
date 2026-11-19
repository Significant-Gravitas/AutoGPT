/** One-click repeat times for a workflow, as cron in the person's zone. */
export const QUICK_SCHEDULES = [
  { label: "Every hour", cron: "0 * * * *" },
  { label: "Every day 9:00", cron: "0 9 * * *" },
  { label: "Every Monday 9:00", cron: "0 9 * * 1" },
] as const;

export const DEFAULT_CUSTOM_TIME = "09:00";

/** "HH:MM" every day, as cron; null for a time that does not read. */
export function dailyCron(time: string) {
  const match = /^(\d{1,2}):(\d{2})$/.exec(time.trim());
  if (!match) return null;
  const hour = Number(match[1]);
  const minute = Number(match[2]);
  if (hour > 23 || minute > 59) return null;
  return `${minute} ${hour} * * *`;
}

export function browserTimezone() {
  return Intl.DateTimeFormat().resolvedOptions().timeZone;
}
