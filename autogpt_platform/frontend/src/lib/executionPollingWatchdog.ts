import { AgentExecutionStatus } from "@/app/api/__generated__/models/agentExecutionStatus";

export const EMPTY_EXECUTION_UPDATES_THRESHOLD = 40;

const POLLING_STATUSES = new Set<AgentExecutionStatus>([
  AgentExecutionStatus.RUNNING,
  AgentExecutionStatus.QUEUED,
  AgentExecutionStatus.INCOMPLETE,
  AgentExecutionStatus.REVIEW,
]);

export function isEmptyExecutionUpdate(rawData: unknown): boolean {
  if (!rawData || typeof rawData !== "object" || Array.isArray(rawData))
    return true;
  if (!("status" in rawData) || (rawData as { status: unknown }).status !== 200)
    return false;
  const payload = (rawData as { data?: unknown }).data;
  if (!payload || typeof payload !== "object" || Array.isArray(payload))
    return true;
  const record = payload as Record<string, unknown>;
  const status = record.status;
  if (
    typeof status !== "string" ||
    !POLLING_STATUSES.has(status as AgentExecutionStatus)
  )
    return true;
  const nodeExecutions = record.node_executions;
  if (!Array.isArray(nodeExecutions)) return true;
  return nodeExecutions.length === 0;
}

export function isPollingStatus(
  status: string | undefined,
): status is AgentExecutionStatus {
  if (!status) return false;
  return POLLING_STATUSES.has(status as AgentExecutionStatus);
}
