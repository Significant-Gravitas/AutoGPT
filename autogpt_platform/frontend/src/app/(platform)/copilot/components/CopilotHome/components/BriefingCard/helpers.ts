import { format, isToday } from "date-fns";

export function formatBriefingDate(date: Date | string): string {
  const parsed = new Date(date);
  if (isToday(parsed)) return "This morning";
  return format(parsed, "MMMM d");
}
