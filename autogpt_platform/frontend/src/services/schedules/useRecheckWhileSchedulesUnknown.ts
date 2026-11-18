import { useRef } from "react";
import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import {
  hasUnknownSchedules,
  SCHEDULER_RETRY_AFTER_MS,
} from "./scheduler-availability";

// A deploy leaves no scheduler for about 1-2.5 min. A library list rechecks
// every loaded page at once, so past that it rechecks once a minute.
const DEPLOY_GAP_MS = 150_000;
const LONG_OUTAGE_RECHECK_MS = 60_000;

// refetchInterval for a library agent list: recheck after Retry-After while
// its schedules are unknown, less often once the outage outlasts a deploy,
// and not at all once they are known.
export function useRecheckWhileSchedulesUnknown() {
  const unknownSince = useRef<number | null>(null);

  return function recheck(agents: LibraryAgent[]) {
    if (!hasUnknownSchedules(agents)) {
      unknownSince.current = null;
      return false;
    }
    unknownSince.current ??= Date.now();
    return Date.now() - unknownSince.current < DEPLOY_GAP_MS
      ? SCHEDULER_RETRY_AFTER_MS
      : LONG_OUTAGE_RECHECK_MS;
  };
}
