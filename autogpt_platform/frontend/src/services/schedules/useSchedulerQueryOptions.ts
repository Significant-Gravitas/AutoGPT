import { useQueryClient } from "@tanstack/react-query";
import {
  isSchedulerUnavailableError,
  recheckWhileSchedulerUnavailable,
} from "./scheduler-availability";

// React Query's own default when the client sets none.
const DEFAULT_RETRIES = 3;

interface QueryWithError {
  state: { error: unknown };
}

function unlessSchedulerUnavailable(query: QueryWithError) {
  return !isSchedulerUnavailableError(query.state.error);
}

// Options for every observer of a schedule list. A scheduler 503 asks to be
// retried after Retry-After, so it isn't retried at once, nor refetched on
// focus or reconnect; Try again still refetches. Any other error keeps the
// client's retry count.
export function useSchedulerRetryOptions() {
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

  return {
    retry,
    refetchOnWindowFocus: unlessSchedulerUnavailable,
    refetchOnReconnect: unlessSchedulerUnavailable,
  };
}

// The same, plus the recheck after Retry-After. One observer per list owns the
// recheck, so the list isn't asked twice per interval.
export function useSchedulerQueryOptions() {
  return {
    ...useSchedulerRetryOptions(),
    refetchInterval: recheckWhileSchedulerUnavailable,
  };
}
