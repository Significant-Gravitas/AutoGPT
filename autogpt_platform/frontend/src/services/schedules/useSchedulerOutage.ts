import { useEffect, useState } from "react";
import { isSchedulerUnavailableError } from "./scheduler-availability";

interface ScheduleQuery {
  error: unknown;
  isError: boolean;
  isSuccess: boolean;
}

// Whether a schedule list is out because the scheduler is restarting. A query
// that never loaded drops its error as soon as it refetches, so without the
// last answer each recheck would flash the loading state until it is back.
export function useSchedulerOutage(query: ScheduleQuery) {
  const [lastOutage, setLastOutage] = useState(false);
  const outage = query.isError && isSchedulerUnavailableError(query.error);
  const settled = query.isError || query.isSuccess;

  useEffect(() => {
    if (settled) setLastOutage(outage);
  }, [settled, outage]);

  return outage || (lastOutage && !settled);
}
