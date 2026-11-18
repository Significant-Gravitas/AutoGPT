import { useQueryClient } from "@tanstack/react-query";
import {
  isSchedulerUnavailableError,
  recheckWhileSchedulerUnavailable,
} from "./scheduler-availability";

// React Query's own default when the client sets none.
const DEFAULT_RETRIES = 3;

// Query options for a schedule list. A scheduler 503 isn't retried at once:
// the backend asks to be retried after Retry-After, and refetchInterval
// rechecks then. Any other error keeps the client's retry count.
export function useSchedulerQueryOptions() {
  const fallback = useQueryClient().getDefaultOptions().queries?.retry;
  const maxRetries =
    typeof fallback === "number"
      ? fallback
      : fallback === false
        ? 0
        : DEFAULT_RETRIES;

  function retry(failureCount: number, error: unknown) {
    return !isSchedulerUnavailableError(error) && failureCount < maxRetries;
  }

  return { retry, refetchInterval: recheckWhileSchedulerUnavailable };
}
