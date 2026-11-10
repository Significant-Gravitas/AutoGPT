import type { getV1GetExecutionDetailsResponse } from "@/app/api/__generated__/endpoints/graphs/graphs";
import type { GraphExecution } from "@/app/api/__generated__/models/graphExecution";
import { ApiError } from "@/lib/autogpt-server-api/helpers";

export const LIVE_STATUS_POLL_MS = 5000;
const MAX_POLL_MS = 60_000;
const POLLS_PER_STEP = 12;
const PERMANENT_ERRORS = new Set([400, 403, 404, 422]);

const FINAL_STATUSES = new Set(["COMPLETED", "FAILED", "TERMINATED"]);
const STOPPED_WITH_ERROR = new Set(["FAILED", "TERMINATED"]);

export function isFinalStatus(status: string | undefined) {
  return FINAL_STATUSES.has((status ?? "").toUpperCase());
}

function isPermanentError(error: unknown) {
  return error instanceof ApiError && PERMANENT_ERRORS.has(error.status);
}

export function retryUnlessPermanent(failureCount: number, error: unknown) {
  return failureCount < 3 && !isPermanentError(error);
}

export function shouldWatchRun(status: string | undefined) {
  // A scheduled run's id belongs to the schedule, not to an execution.
  return !isFinalStatus(status) && (status ?? "").toUpperCase() !== "SCHEDULED";
}

interface PollState {
  data: getV1GetExecutionDetailsResponse | undefined;
  error: unknown;
  dataUpdateCount: number;
  errorUpdateCount: number;
}

function backedOffInterval(polls: number) {
  // 5 s for the first minute, then twice as long each step, up to a minute,
  // so a run parked in review for hours doesn't keep fetching every 5 s.
  const step = Math.floor(polls / POLLS_PER_STEP);
  return Math.min(LIVE_STATUS_POLL_MS * 2 ** step, MAX_POLL_MS);
}

export function nextPollInterval(state: PollState) {
  const { data, error } = state;
  // A run that has ended stays ended, even if a later refetch fails.
  if (data?.status === 200 && isFinalStatus(data.data.status)) return false;
  // These won't fix themselves (the run is gone or not ours). A 401 during a
  // session refresh, a 5xx or a network error might, so those keep polling.
  if (isPermanentError(error)) return false;
  if (!error && data?.status !== 200) return false;
  return backedOffInterval(state.dataUpdateCount + state.errorUpdateCount);
}

export function runErrorFor(run: Pick<GraphExecution, "status" | "stats">) {
  if (!STOPPED_WITH_ERROR.has(run.status)) return null;
  return run.stats?.error?.trim() || null;
}
