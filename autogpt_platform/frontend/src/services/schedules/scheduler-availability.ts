import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { ApiError } from "@/lib/autogpt-server-api/helpers";

// The backend's Retry-After on a scheduler 503 (SCHEDULER_RETRY_AFTER_SECONDS
// in backend/api/utils/exceptions.py). A deploy leaves no scheduler for about
// 1-2.5 min, so a recheck this often lands shortly after it is back.
export const SCHEDULER_RETRY_AFTER_MS = 15_000;

// The scheduler restarts on every deploy and answers 503 while it is out: the
// schedules are unknown for a moment, neither missing nor broken.
export function isSchedulerUnavailableError(error: unknown) {
  return error instanceof ApiError && error.status === 503;
}

// Library reads still load without the scheduler; their agents then carry
// their schedules as unknown rather than absent.
export function hasUnknownSchedules(agents: LibraryAgent[]) {
  return agents.some((agent) => agent.schedule_info_unavailable);
}

// refetchInterval for a schedule list: recheck after Retry-After while the
// scheduler is out, and not at all once it answers. React Query pauses the
// interval while the tab is hidden.
export function recheckWhileSchedulerUnavailable(query: {
  state: { error: unknown };
}) {
  return isSchedulerUnavailableError(query.state.error)
    ? SCHEDULER_RETRY_AFTER_MS
    : false;
}
