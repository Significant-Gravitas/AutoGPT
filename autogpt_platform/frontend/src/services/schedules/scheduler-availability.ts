import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { ApiError } from "@/lib/autogpt-server-api/helpers";

// Matches the backend's Retry-After on a scheduler 503.
export const SCHEDULER_RECHECK_MS = 15_000;
const SCHEDULER_RECHECK_MAX_MS = 120_000;

type SchedulerErrorCode = "scheduler_unavailable" | "scheduler_unconfirmed";

function schedulerErrorCode(error: unknown): SchedulerErrorCode | null {
  if (!(error instanceof ApiError) || error.status !== 503) return null;
  const code: unknown = error.response?.detail?.code;
  if (code === "scheduler_unavailable" || code === "scheduler_unconfirmed") {
    return code;
  }
  return null;
}

// The scheduler restarts on every deploy. Its 503 means "try again shortly",
// not "you have no schedules", and must never be shown as either an error or
// an empty list.
export function isSchedulerUnavailableError(error: unknown): boolean {
  return schedulerErrorCode(error) !== null;
}

// A write reached the scheduler but no answer came back, so it may have been
// applied. The caller should re-read before offering a retry.
export function isSchedulerUnconfirmedError(error: unknown): boolean {
  return schedulerErrorCode(error) === "scheduler_unconfirmed";
}

export function hasUnavailableScheduleInfo(agents: LibraryAgent[]): boolean {
  return agents.some((agent) => agent.schedule_info_unavailable);
}

// Schedule queries recheck until the scheduler answers again, backing off from
// 15s to 2 min so an open tab doesn't poll a long outage every 15s.
export function recheckWhileSchedulerUnavailable<TError>(query: {
  state: { error: TError; errorUpdateCount: number };
}): number | false {
  if (!isSchedulerUnavailableError(query.state.error)) return false;
  return schedulerRecheckDelay(query.state.errorUpdateCount);
}

export function schedulerRecheckDelay(failures: number): number {
  const doublings = Math.max(0, failures - 1);
  return Math.min(
    SCHEDULER_RECHECK_MS * 2 ** doublings,
    SCHEDULER_RECHECK_MAX_MS,
  );
}
