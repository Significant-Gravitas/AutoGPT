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

// e.g. "June 26" (current year) or "December 1, 2024" (older), in the user's
// locale. A single Intl.DateTimeFormat options set keeps the format valid for
// every locale instead of hand-assembling English-style date parts.
function formatDayLabel(date: Date, locale?: string): string {
  const sameYear = date.getFullYear() === new Date().getFullYear();
  return new Intl.DateTimeFormat(locale, {
    day: "numeric",
    month: "long",
    ...(sameYear ? {} : { year: "numeric" }),
  }).format(date);
}

export function getDateGroupLabel(iso: string, locale?: string): string {
  const diffDays = diffInDays(iso);
  if (diffDays <= 0) return "Today";
  if (diffDays === 1) return "Yesterday";
  return formatDayLabel(new Date(iso), locale);
}

// Buckets sessions by calendar day and orders groups most-recent-first. Doesn't
// rely on the input being pre-sorted, so an unsorted list can never produce
// duplicate day groups; within a group, input order is preserved.
export function groupSessionsByDate<T extends { updated_at: string }>(
  sessions: T[],
  locale?: string,
): { label: string; sessions: T[] }[] {
  const buckets = new Map<number, { label: string; sessions: T[] }>();

  for (const session of sessions) {
    const dayKey = startOfDay(new Date(session.updated_at));
    const bucket = buckets.get(dayKey);
    if (bucket) {
      bucket.sessions.push(session);
    } else {
      buckets.set(dayKey, {
        label: getDateGroupLabel(session.updated_at, locale),
        sessions: [session],
      });
    }
  }

  return [...buckets.entries()]
    .sort((a, b) => b[0] - a[0])
    .map(([, group]) => group);
}
