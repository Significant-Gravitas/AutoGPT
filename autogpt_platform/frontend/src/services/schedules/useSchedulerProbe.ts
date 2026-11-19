import { useEffect, useRef, useState } from "react";
import { useGetV1ListExecutionSchedulesForAUser } from "@/app/api/__generated__/endpoints/schedules/schedules";
import { useRecheckWhileSchedulesUnknown } from "./useRecheckWhileSchedulesUnknown";
import { useSchedulerRetryOptions } from "./useSchedulerQueryOptions";

// While a list's schedules are unknown, probe the scheduler with one read
// — the user's schedule list — instead of re-reading every loaded page of
// the list. The probe is one query for every caller, so it is fetched once
// per interval no matter how many lists use it. `recovered` is true once a
// probe has answered after the schedules went unknown: the caller re-reads
// its list once, and it goes false again as soon as that re-read shows
// known schedules. A cached success from before the outage does not count:
// the probe query keeps its last answer while disabled, so without this
// the list would re-read its pages at the very start of the outage,
// before the new probe has answered. A success stays recorded through later
// probe errors for the same unknown period, so a flapping probe cannot
// re-trigger the one re-read; only known schedules clear it.
export function useSchedulerProbe(schedulesUnknown: boolean) {
  const recheck = useRecheckWhileSchedulesUnknown();
  const retryOptions = useSchedulerRetryOptions();
  const unknownSince = useRef<number | null>(null);
  const [recovered, setRecovered] = useState(false);

  useEffect(() => {
    if (schedulesUnknown) {
      unknownSince.current ??= Date.now();
    } else {
      unknownSince.current = null;
      setRecovered(false);
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
  // leave later probe updates unobserved and `recovered` stuck false when
  // the hook mounts with a cached success.
  const dataUpdatedAt = probe.dataUpdatedAt;
  const probeRecovered =
    schedulesUnknown &&
    probe.isSuccess &&
    unknownSince.current !== null &&
    dataUpdatedAt >= unknownSince.current;

  useEffect(() => {
    if (probeRecovered) setRecovered(true);
  }, [probeRecovered]);

  return { recovered };
}
