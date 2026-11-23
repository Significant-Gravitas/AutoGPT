import { useNowMinute } from "@/hooks/useNowMinute";
import { formatTimeAgo } from "@/lib/utils/time";
import { formatDistanceToNow } from "date-fns";
import { useSyncExternalStore } from "react";

export type RelativeTimeVariant = "long" | "short";

// Reads neither the clock nor the local time zone, so the server and the
// browser write the same text.
const DATE_FORMAT = new Intl.DateTimeFormat("en-US", {
  dateStyle: "medium",
  timeZone: "UTC",
});

function subscribeToNothing() {
  return () => {};
}

/**
 * `date` relative to now: "5 minutes ago", or "5m ago" for the short variant.
 * Null when there is no date to read.
 *
 * The server's clock and the visitor's never agree, so the HTML the server
 * sends, and the render that hydrates it, show the date itself instead
 * ("Apr 23, 2026", in UTC); the relative time follows straight after
 * hydration. A component that mounts later shows it from its first render.
 *
 * The date is in UTC, not the visitor's time zone, because the server
 * doesn't know that zone: a zone-dependent date is a hydration mismatch of
 * its own. Far from UTC, it can read as the next or previous day until the
 * page hydrates.
 * It re-renders on the shared minute clock, so it keeps up with time.
 */
export function useRelativeTime(
  date: Date | string | number | null | undefined,
  variant: RelativeTimeVariant = "long",
) {
  const isHydrated = useSyncExternalStore(
    subscribeToNothing,
    () => true,
    () => false,
  );
  useNowMinute(isHydrated);

  if (date === null || date === undefined) return null;
  const parsed = new Date(date);
  if (Number.isNaN(parsed.getTime())) return null;

  if (!isHydrated) return DATE_FORMAT.format(parsed);
  if (variant === "short") return formatTimeAgo(parsed);
  return formatDistanceToNow(parsed, { addSuffix: true });
}
