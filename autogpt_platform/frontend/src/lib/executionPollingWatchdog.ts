import { AgentExecutionStatus } from "@/app/api/__generated__/models/agentExecutionStatus";

export const EMPTY_EXECUTION_UPDATES_THRESHOLD = 40;

const POLLING_STATUSES = new Set<AgentExecutionStatus>([
  AgentExecutionStatus.RUNNING,
  AgentExecutionStatus.QUEUED,
  AgentExecutionStatus.INCOMPLETE,
  AgentExecutionStatus.REVIEW,
]);

export function isEmptyExecutionUpdate(rawData: unknown): boolean {
  if (!rawData || typeof rawData !== "object" || !("status" in rawData))
    return true;
  if ((rawData as { status: unknown }).status !== 200) return false;
  const data = (rawData as { data?: unknown }).data;
  if (!data || typeof data !== "object" || Array.isArray(data)) return true;
  const status = (data as { status?: string }).status;
  if (!status || !POLLING_STATUSES.has(status)) return false;
  const nodeExecutions = (data as { node_executions?: unknown[] })
    .node_executions;
  return !Array.isArray(nodeExecutions) || nodeExecutions.length === 0;
}

export function isPollingStatus(status: string | undefined): boolean {
  return !!status && POLLING_STATUSES.has(status);
}
