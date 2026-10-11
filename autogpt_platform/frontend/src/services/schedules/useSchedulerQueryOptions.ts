import { useQueryClient } from "@tanstack/react-query";
import {
  isSchedulerUnavailableError,
  recheckWhileSchedulerUnavailable,
} from "./scheduler-availability";

const REACT_QUERY_DEFAULT_RETRIES = 3;

// Query options for schedule lists. A scheduler 503 is not retried, since it is
// restarting and rechecks come from refetchInterval. Every other error keeps
// the client's own retry setting.
export function useSchedulerQueryOptions() {
  const fallback = useQueryClient().getDefaultOptions().queries?.retry;
  const maxRetries =
    typeof fallback === "number"
      ? fallback
      : fallback === false
        ? 0
        : REACT_QUERY_DEFAULT_RETRIES;

  return {
    retry: <TError>(failureCount: number, error: TError) =>
      !isSchedulerUnavailableError(error) && failureCount < maxRetries,
    refetchInterval: recheckWhileSchedulerUnavailable,
  };
}
