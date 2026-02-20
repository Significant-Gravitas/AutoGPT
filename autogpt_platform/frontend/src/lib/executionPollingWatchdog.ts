/** ~60s at 1.5s polling interval; used to detect no-progress and pause polling. */
export const EMPTY_EXECUTION_UPDATES_THRESHOLD = 40;

/** Shared polling interval (ms) for execution details when in polling state. */
export const EXECUTION_POLLING_INTERVAL_MS = 1500;

const POLLING_STATUSES = new Set<string>([
  "RUNNING",
  "QUEUED",
  "INCOMPLETE",
  "REVIEW",
]);

const TERMINAL_STATUSES = new Set<string>([
  "COMPLETED",
  "FAILED",
  "TERMINATED",
]);

/**
 * Returns true only when the response is a valid 200 with a polling status
 * and node_executions is present as an empty array (no progress).
 * Malformed payloads and non-polling states are not classified as empty.
 */
export function isEmptyExecutionUpdate(rawData: unknown): boolean {
  if (!rawData || typeof rawData !== "object" || Array.isArray(rawData))
    return false;
  if (!("status" in rawData) || (rawData as { status: unknown }).status !== 200)
    return false;
  const payload = (rawData as { data?: unknown }).data;
  if (!payload || typeof payload !== "object" || Array.isArray(payload))
    return false;
  const record = payload as Record<string, unknown>;
  const status = record.status;
  if (typeof status !== "string") return false;
  if (TERMINAL_STATUSES.has(status)) return false;
  if (!POLLING_STATUSES.has(status)) return false;
  const nodeExecutions = record.node_executions;
  if (!Array.isArray(nodeExecutions)) return false;
  return nodeExecutions.length === 0;
}

export function isPollingStatus(status: string | undefined): boolean {
  if (!status) return false;
  return POLLING_STATUSES.has(status);
}
