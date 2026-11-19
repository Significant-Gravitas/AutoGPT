import { useGetV1ListExecutionSchedulesForAUser } from "@/app/api/__generated__/endpoints/schedules/schedules";
import { useRecheckWhileSchedulesUnknown } from "./useRecheckWhileSchedulesUnknown";
import { useSchedulerRetryOptions } from "./useSchedulerQueryOptions";

// While a list's schedules are unknown, probe the scheduler with one read
// — the user's schedule list — instead of re-reading every loaded page of
// the list. The probe is one query for every caller, so it is fetched once
// per interval no matter how many lists use it. `recovered` is true while
// the probe has answered and the list's schedules are still unknown: the
// caller re-reads its list once, and it goes false again as soon as that
// re-read shows known schedules.
export function useSchedulerProbe(schedulesUnknown: boolean) {
  const recheck = useRecheckWhileSchedulesUnknown();
  const retryOptions = useSchedulerRetryOptions();
  const probe = useGetV1ListExecutionSchedulesForAUser({
    query: {
      ...retryOptions,
      refetchInterval: () => recheck(schedulesUnknown),
      enabled: schedulesUnknown,
    },
  });
  return { recovered: schedulesUnknown && probe.isSuccess };
}
