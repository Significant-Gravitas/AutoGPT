import { format, isToday, parseISO } from "date-fns";

export function formatBriefingDate(date: Date | string): string {
  // The generated type says Date, but date-only strings ("2026-08-07") skip
  // the client's date transformer (its regex requires a time part), so the
  // runtime value is still a string. parseISO reads it as LOCAL midnight —
  // `new Date("2026-08-07")` would read UTC midnight and shift the label a
  // day back for viewers west of UTC.
  const parsed = typeof date === "string" ? parseISO(date) : date;
  if (isToday(parsed)) return "This morning";
  return format(parsed, "MMMM d");
}

export function isInternalLink(link: string): boolean {
  return link.startsWith("/") && !link.startsWith("//");
}
