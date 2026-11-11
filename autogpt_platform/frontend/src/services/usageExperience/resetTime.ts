const MINUTE_MS = 60_000;
const DAY_MS = 24 * 60 * MINUTE_MS;

/** A usage reset can fall on any second (each customer's own billing clock),
 *  and is shown to the minute: rounded up, so the time on screen is never
 *  before the reset itself. */
export function ceilToMinute(date: Date) {
  return new Date(Math.ceil(date.getTime() / MINUTE_MS) * MINUTE_MS);
}

/** A weekly window can be 7 days 18 hours long, so from six days out a
 *  weekday alone could mean this week or next: the date goes with it. */
export function needsDate(date: Date, now = new Date()) {
  return date.getTime() - now.getTime() >= 6 * DAY_MS;
}

/** "Thu, Nov 12" in the reader's locale. */
export function formatWeekdayDate(date: Date) {
  return date.toLocaleDateString(undefined, {
    weekday: "short",
    month: "short",
    day: "numeric",
  });
}

/** The exact reset moment in the reader's own time zone: "Thu, Nov 12,
 *  8:31 AM". Null for a missing or unreadable time. */
export function formatExactReset(value: Date | string | null | undefined) {
  if (!value) return null;
  const date = new Date(value);
  if (!Number.isFinite(date.getTime())) return null;
  return (
    ceilToMinute(date)
      .toLocaleString(undefined, {
        weekday: "short",
        month: "short",
        day: "numeric",
        hour: "numeric",
        minute: "2-digit",
      })
      // Some ICU versions put a narrow no-break space before AM/PM.
      .replace(/ /g, " ")
  );
}
