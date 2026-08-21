function startOfDay(date: Date): number {
  return new Date(
    date.getFullYear(),
    date.getMonth(),
    date.getDate(),
  ).getTime();
}

function diffInDays(iso: string): number {
  const date = new Date(iso);
  const dayMs = 86_400_000;
  return Math.round((startOfDay(new Date()) - startOfDay(date)) / dayMs);
}

function formatDayLabel(date: Date): string {
  const sameYear = date.getFullYear() === new Date().getFullYear();
  const options: Intl.DateTimeFormatOptions = sameYear
    ? { month: "long", day: "numeric" }
    : { month: "long", day: "numeric", year: "numeric" };
  return new Intl.DateTimeFormat(undefined, options).format(date);
}

export function getDateGroupLabel(iso: string): string {
  const diffDays = diffInDays(iso);
  if (diffDays <= 0) return "Today";
  if (diffDays === 1) return "Yesterday";
  return formatDayLabel(new Date(iso));
}

// Buckets sessions by calendar day and orders groups most-recent-first. Doesn't
// rely on the input being pre-sorted, so an unsorted list can never produce
// duplicate day groups; within a group, input order is preserved.
export function groupSessionsByDate<T extends { updated_at: string }>(
  sessions: T[],
): { label: string; sessions: T[] }[] {
  const buckets = new Map<number, { label: string; sessions: T[] }>();

  for (const session of sessions) {
    const dayKey = startOfDay(new Date(session.updated_at));
    const bucket = buckets.get(dayKey);
    if (bucket) {
      bucket.sessions.push(session);
    } else {
      buckets.set(dayKey, {
        label: getDateGroupLabel(session.updated_at),
        sessions: [session],
      });
    }
  }

  return [...buckets.entries()]
    .sort((a, b) => b[0] - a[0])
    .map(([, group]) => group);
}
