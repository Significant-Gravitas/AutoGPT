import { useRef } from "react";
import { SCHEDULER_RETRY_AFTER_MS } from "./scheduler-availability";

// A deploy leaves no scheduler for about 1-2.5 min. A library list rechecks
// every loaded page at once, and Home its whole dashboard, so past that they
// recheck once a minute.
const DEPLOY_GAP_MS = 150_000;
const LONG_OUTAGE_RECHECK_MS = 60_000;

// refetchInterval for a read that answers with its schedules unknown: recheck
// after Retry-After, less often once the outage outlasts a deploy, and not at
// all once they are known.
export function useRecheckWhileSchedulesUnknown() {
  const unknownSince = useRef<number | null>(null);

  return function recheck(schedulesUnknown: boolean) {
    if (!schedulesUnknown) {
      unknownSince.current = null;
      return false;
    }
    unknownSince.current ??= Date.now();
    return Date.now() - unknownSince.current < DEPLOY_GAP_MS
      ? SCHEDULER_RETRY_AFTER_MS
      : LONG_OUTAGE_RECHECK_MS;
  };
}
