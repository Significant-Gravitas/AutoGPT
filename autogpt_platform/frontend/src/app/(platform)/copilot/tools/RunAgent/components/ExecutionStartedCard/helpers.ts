import type { getV1GetExecutionDetailsResponse } from "@/app/api/__generated__/endpoints/graphs/graphs";
import type { GraphExecution } from "@/app/api/__generated__/models/graphExecution";

export const LIVE_STATUS_POLL_MS = 5000;

const FINAL_STATUSES = new Set(["COMPLETED", "FAILED", "TERMINATED"]);
const STOPPED_WITH_ERROR = new Set(["FAILED", "TERMINATED"]);

export function isFinalStatus(status: string | undefined) {
  return FINAL_STATUSES.has((status ?? "").toUpperCase());
}

export function shouldWatchRun(status: string | undefined) {
  // A scheduled run's id belongs to the schedule, not to an execution.
  return !isFinalStatus(status) && (status ?? "").toUpperCase() !== "SCHEDULED";
}

export function nextPollInterval(
  response: getV1GetExecutionDetailsResponse | undefined,
) {
  if (response?.status !== 200) return false;
  return isFinalStatus(response.data.status) ? false : LIVE_STATUS_POLL_MS;
}

export function runErrorFor(run: Pick<GraphExecution, "status" | "stats">) {
  if (!STOPPED_WITH_ERROR.has(run.status)) return null;
  return run.stats?.error?.trim() || null;
}
