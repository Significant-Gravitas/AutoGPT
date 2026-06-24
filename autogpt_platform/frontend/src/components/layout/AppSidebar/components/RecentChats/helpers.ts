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

export function formatChatDate(iso: string): string {
  const diffDays = diffInDays(iso);
  if (diffDays <= 0) return "Today";
  if (diffDays === 1) return "Yesterday";

  const date = new Date(iso);
  const sameYear = date.getFullYear() === new Date().getFullYear();
  return date.toLocaleDateString(undefined, {
    month: "short",
    day: "numeric",
    year: sameYear ? undefined : "numeric",
  });
}

const DATE_GROUP_ORDER = [
  "Today",
  "Yesterday",
  "Previous 7 days",
  "Previous 30 days",
  "Older",
] as const;

export function getDateGroupLabel(
  iso: string,
): (typeof DATE_GROUP_ORDER)[number] {
  const diffDays = diffInDays(iso);
  if (diffDays <= 0) return "Today";
  if (diffDays === 1) return "Yesterday";
  if (diffDays <= 7) return "Previous 7 days";
  if (diffDays <= 30) return "Previous 30 days";
  return "Older";
}

// Buckets sessions into a fixed chronological group order. Doesn't rely on the
// input being pre-sorted, so an unsorted list can never produce duplicate
// labels (e.g. "Today" … "Today"); within a group, input order is preserved.
export function groupSessionsByDate<T extends { updated_at: string }>(
  sessions: T[],
): { label: string; sessions: T[] }[] {
  const buckets = new Map<string, T[]>();

  for (const session of sessions) {
    const label = getDateGroupLabel(session.updated_at);
    const bucket = buckets.get(label);
    if (bucket) {
      bucket.push(session);
    } else {
      buckets.set(label, [session]);
    }
  }

  return DATE_GROUP_ORDER.filter((label) => buckets.has(label)).map(
    (label) => ({
      label,
      sessions: buckets.get(label) ?? [],
    }),
  );
}
