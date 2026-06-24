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

export function getDateGroupLabel(iso: string): string {
  const diffDays = diffInDays(iso);
  if (diffDays <= 0) return "Today";
  if (diffDays === 1) return "Yesterday";
  if (diffDays <= 7) return "Previous 7 days";
  if (diffDays <= 30) return "Previous 30 days";
  return "Older";
}

export function groupSessionsByDate<T extends { updated_at: string }>(
  sessions: T[],
): { label: string; sessions: T[] }[] {
  const groups: { label: string; sessions: T[] }[] = [];

  for (const session of sessions) {
    const label = getDateGroupLabel(session.updated_at);
    const lastGroup = groups[groups.length - 1];

    if (lastGroup && lastGroup.label === label) {
      lastGroup.sessions.push(session);
    } else {
      groups.push({ label, sessions: [session] });
    }
  }

  return groups;
}
