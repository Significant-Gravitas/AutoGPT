import type { SessionSummaryResponse } from "@/app/api/__generated__/models/sessionSummaryResponse";
import { useEffect, useState } from "react";

const DAY_MS = 24 * 60 * 60 * 1000;
// `ChatSessionBlock` labels a session "Today", "Yesterday" or "N days ago"
// from whole 24-hour periods since `updated_at`, and shows a date from 7 on.
const LAST_RELATIVE_DAY = 7;

/**
 * Whole days since `updatedAt`, capped where the label turns into a date. It
 * changes exactly when the row's relative date label does, so passing it to a
 * memoised row re-renders that row then and at no other time.
 */
export function sessionAgeDays(updatedAt: string | Date, now: number) {
  const days = Math.floor((now - new Date(updatedAt).getTime()) / DAY_MS);
  return Math.min(days, LAST_RELATIVE_DAY);
}

/**
 * A timestamp that moves on only when some session's age in days changes:
 * one timeout, set for the earliest such moment across the list.
 */
export function useSessionAgeClock(sessions: SessionSummaryResponse[]) {
  const [now, setNow] = useState(() => Date.now());

  useEffect(() => {
    const current = Date.now();
    let next = Infinity;
    for (const session of sessions) {
      const updated = new Date(session.updated_at).getTime();
      const days = Math.floor((current - updated) / DAY_MS);
      if (days >= LAST_RELATIVE_DAY) continue;
      next = Math.min(next, updated + (days + 1) * DAY_MS);
    }
    if (next === Infinity) return;
    const timeout = setTimeout(() => setNow(Date.now()), next - current + 1);
    return () => clearTimeout(timeout);
  }, [sessions, now]);

  return now;
}
