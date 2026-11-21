import { useEffect, useRef } from "react";
import { useGetV1ListExecutionSchedulesForAUser } from "@/app/api/__generated__/endpoints/schedules/schedules";
import { useRecheckWhileSchedulesUnknown } from "./useRecheckWhileSchedulesUnknown";
import { useSchedulerRetryOptions } from "./useSchedulerQueryOptions";

// While a list's schedules are unknown, probe the scheduler with one read —
// the user's schedule list — instead of re-reading every loaded page.
// `recovered` is the time of the latest probe answer since they went unknown
// (a cached answer from before does not count), so the caller re-reads once
// per answer until its schedules are known again. A re-read that still finds
// schedules unknown is retried on the next answer: the probe's read waits
// longer for the scheduler than a page read does, so the first re-read after
// a restart can still meet unknown schedules while the probe keeps
// answering. While the scheduler is out the probe errors and answers
// nothing, so no list re-reads and the saving this probe is after holds.
export function useSchedulerProbe(schedulesUnknown: boolean) {
  const recheck = useRecheckWhileSchedulesUnknown();
  const retryOptions = useSchedulerRetryOptions();
  const unknownSince = useRef<number | null>(null);

  useEffect(() => {
    if (schedulesUnknown) {
      unknownSince.current ??= Date.now();
    } else {
      unknownSince.current = null;
    }
  }, [schedulesUnknown]);

  const probe = useGetV1ListExecutionSchedulesForAUser({
    query: {
      ...retryOptions,
      refetchInterval: () => recheck(schedulesUnknown),
      enabled: schedulesUnknown,
    },
  });

  // Read dataUpdatedAt unconditionally: the query result only notifies this
  // hook about properties the render reads, so a short-circuited read would
  // leave later probe updates unobserved and `recovered` stuck null when
  // the hook mounts with a cached success.
  const dataUpdatedAt = probe.dataUpdatedAt;
  const recovered =
    schedulesUnknown &&
    probe.isSuccess &&
    unknownSince.current !== null &&
    dataUpdatedAt >= unknownSince.current
      ? dataUpdatedAt
      : null;

  return { recovered };
}
