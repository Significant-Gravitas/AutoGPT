const DAY = 24 * 60 * 60 * 1000;

function startOfDay(date: Date) {
  return new Date(date.getFullYear(), date.getMonth(), date.getDate()).getTime();
}

/** A chat-list timestamp: the time today, then Yesterday, a weekday, a date. */
export function shortTime(iso: string, now = new Date()): string | null {
  const then = new Date(iso);
  if (Number.isNaN(then.getTime())) return null;
  // Rounded, so a daylight-saving day of 23 or 25 hours still counts as one.
  const days = Math.round((startOfDay(now) - startOfDay(then)) / DAY);
  if (days === 0) {
    return then.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
  }
  if (days === 1) return "Yesterday";
  if (days > 1 && days < 7) {
    return then.toLocaleDateString([], { weekday: "short" });
  }
  return `${then.getDate()} ${then.toLocaleDateString([], { month: "short" })}`;
}
