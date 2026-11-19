import { useEffect, useRef } from "react";
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
// before the new probe has answered.
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
  const recovered =
    schedulesUnknown &&
    probe.isSuccess &&
    unknownSince.current !== null &&
    probe.dataUpdatedAt >= unknownSince.current;
  return { recovered };
}
