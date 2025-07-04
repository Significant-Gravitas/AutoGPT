import { AgentExecutionStatus } from "@/app/api/__generated__/models/agentExecutionStatus";
import { GraphExecutionMeta as GeneratedGraphExecutionMeta } from "@/app/api/__generated__/models/graphExecutionMeta";
import { MyAgent } from "@/app/api/__generated__/models/myAgent";
import type { GraphExecution } from "@/lib/autogpt-server-api/types";

export function formatTimeAgo(dateStr: string): string {
  const date = new Date(dateStr);
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffMins = Math.floor(diffMs / 60000);

  if (diffMins < 1) return "Just now";
  if (diffMins < 60) return `${diffMins}m ago`;
  const diffHours = Math.floor(diffMins / 60);
  if (diffHours < 24) return `${diffHours}h ago`;
  const diffDays = Math.floor(diffHours / 24);
  return `${diffDays}d ago`;
}

export function getStatusDisplayText(
  execution: GeneratedGraphExecutionMeta,
): string {
  switch (execution.status) {
    case AgentExecutionStatus.QUEUED:
      return "Queued";
    case AgentExecutionStatus.RUNNING:
      return "Running";
    case AgentExecutionStatus.COMPLETED:
      return "Completed";
    case AgentExecutionStatus.FAILED:
      return "Failed";
    case AgentExecutionStatus.TERMINATED:
      return "Stopped";
    case AgentExecutionStatus.INCOMPLETE:
      return "Incomplete";
    default:
      return execution.status;
  }
}

export function getStatusColorClass(
  execution: GeneratedGraphExecutionMeta,
): string {
  switch (execution.status) {
    case AgentExecutionStatus.QUEUED:
      return "text-yellow-600";
    case AgentExecutionStatus.RUNNING:
      return "text-blue-600";
    case AgentExecutionStatus.COMPLETED:
      return "text-green-600";
    case AgentExecutionStatus.FAILED:
    case AgentExecutionStatus.TERMINATED:
      return "text-red-600";
    case AgentExecutionStatus.INCOMPLETE:
      return "text-gray-600";
    default:
      return "text-gray-600";
  }
}

export function truncateGraphId(graphId: string, length: number = 8): string {
  return `${graphId.slice(0, length)}...`;
}

export function getExecutionDuration(
  execution: GeneratedGraphExecutionMeta,
): string {
  if (!execution.started_at) return "Unknown";

  const start = new Date(execution.started_at);
  const end = execution.ended_at ? new Date(execution.ended_at) : new Date();
  const durationMs = end.getTime() - start.getTime();
  const durationSec = Math.floor(durationMs / 1000);

  if (durationSec < 60) return `${durationSec}s`;
  const durationMin = Math.floor(durationSec / 60);
  if (durationMin < 60) return `${durationMin}m ${durationSec % 60}s`;
  const durationHr = Math.floor(durationMin / 60);
  return `${durationHr}h ${durationMin % 60}m`;
}

export function shouldShowNotificationBadge(totalCount: number): boolean {
  return totalCount > 0;
}

export function formatNotificationCount(count: number): string {
  if (count > 99) return "99+";
  return count.toString();
}

export interface AgentExecutionWithInfo extends GeneratedGraphExecutionMeta {
  agent_name: string;
  agent_description: string;
  library_agent_id?: string;
}

export interface NotificationState {
  activeExecutions: AgentExecutionWithInfo[];
  recentCompletions: AgentExecutionWithInfo[];
  recentFailures: AgentExecutionWithInfo[];
  totalCount: number;
}

export function createAgentInfoMap(
  agents: MyAgent[],
): Map<
  string,
  { name: string; description: string; library_agent_id?: string }
> {
  const agentMap = new Map<
    string,
    { name: string; description: string; library_agent_id?: string }
  >();

  agents.forEach((agent) => {
    agentMap.set(agent.agent_id, {
      name: agent.agent_name,
      description: agent.description,
      library_agent_id: undefined, // MyAgent doesn't have library_agent_id
    });
  });

  return agentMap;
}

export function convertLegacyExecutionToGenerated(
  execution: GraphExecution,
): GeneratedGraphExecutionMeta {
  return {
    id: execution.id,
    user_id: execution.user_id,
    graph_id: execution.graph_id,
    graph_version: execution.graph_version,
    preset_id: execution.preset_id,
    status: execution.status as AgentExecutionStatus,
    started_at: execution.started_at.toISOString(),
    ended_at: execution.ended_at.toISOString(),
    stats: execution.stats || {
      cost: 0,
      duration: 0,
      duration_cpu_only: 0,
      node_exec_time: 0,
      node_exec_time_cpu_only: 0,
      node_exec_count: 0,
    },
  };
}

export function enrichExecutionWithAgentInfo(
  execution: GeneratedGraphExecutionMeta,
  agentInfoMap: Map<
    string,
    { name: string; description: string; library_agent_id?: string }
  >,
): AgentExecutionWithInfo {
  const agentInfo = agentInfoMap.get(execution.graph_id);
  return {
    ...execution,
    agent_name: agentInfo?.name || `Graph ${execution.graph_id.slice(0, 8)}...`,
    agent_description: agentInfo?.description ?? "",
    library_agent_id: agentInfo?.library_agent_id,
  };
}

export function isActiveExecution(
  execution: GeneratedGraphExecutionMeta,
): boolean {
  const status = execution.status;
  return (
    status === AgentExecutionStatus.RUNNING ||
    status === AgentExecutionStatus.QUEUED
  );
}

export function isRecentCompletion(
  execution: GeneratedGraphExecutionMeta,
  thirtyMinutesAgo: Date,
): boolean {
  const status = execution.status;
  return (
    status === AgentExecutionStatus.COMPLETED &&
    !!execution.ended_at &&
    new Date(execution.ended_at) > thirtyMinutesAgo
  );
}

export function isRecentFailure(
  execution: GeneratedGraphExecutionMeta,
  thirtyMinutesAgo: Date,
): boolean {
  const status = execution.status;
  return (
    (status === AgentExecutionStatus.FAILED ||
      status === AgentExecutionStatus.TERMINATED) &&
    !!execution.ended_at &&
    new Date(execution.ended_at) > thirtyMinutesAgo
  );
}

export function isRecentNotification(
  execution: AgentExecutionWithInfo,
  thirtyMinutesAgo: Date,
): boolean {
  return execution.ended_at
    ? new Date(execution.ended_at) > thirtyMinutesAgo
    : false;
}

export function categorizeExecutions(
  executions: GeneratedGraphExecutionMeta[],
  agentInfoMap: Map<
    string,
    { name: string; description: string; library_agent_id?: string }
  >,
): NotificationState {
  const twentyFourHoursAgo = new Date(Date.now() - 24 * 60 * 60 * 1000);

  const enrichedExecutions = executions.map((execution) =>
    enrichExecutionWithAgentInfo(execution, agentInfoMap),
  );

  const activeExecutions = enrichedExecutions
    .filter(isActiveExecution)
    .slice(0, 10);

  const recentCompletions = enrichedExecutions
    .filter((execution) => isRecentCompletion(execution, twentyFourHoursAgo))
    .slice(0, 10);

  const recentFailures = enrichedExecutions
    .filter((execution) => isRecentFailure(execution, twentyFourHoursAgo))
    .slice(0, 10);

  return {
    activeExecutions,
    recentCompletions,
    recentFailures,
    totalCount:
      activeExecutions.length +
      recentCompletions.length +
      recentFailures.length,
  };
}

export function removeExecutionFromAllCategories(
  state: NotificationState,
  executionId: string,
): NotificationState {
  return {
    activeExecutions: state.activeExecutions.filter(
      (e) => e.id !== executionId,
    ),
    recentCompletions: state.recentCompletions.filter(
      (e) => e.id !== executionId,
    ),
    recentFailures: state.recentFailures.filter((e) => e.id !== executionId),
    totalCount: state.totalCount, // Will be recalculated later
  };
}

export function addExecutionToCategory(
  state: NotificationState,
  execution: AgentExecutionWithInfo,
): NotificationState {
  const twentyFourHoursAgo = new Date(Date.now() - 24 * 60 * 60 * 1000);
  const newState = { ...state };

  if (isActiveExecution(execution)) {
    newState.activeExecutions = [execution, ...newState.activeExecutions].slice(
      0,
      10,
    );
  } else if (isRecentCompletion(execution, twentyFourHoursAgo)) {
    newState.recentCompletions = [
      execution,
      ...newState.recentCompletions,
    ].slice(0, 10);
  } else if (isRecentFailure(execution, twentyFourHoursAgo)) {
    newState.recentFailures = [execution, ...newState.recentFailures].slice(
      0,
      10,
    );
  }

  return newState;
}

export function cleanupOldNotifications(
  state: NotificationState,
): NotificationState {
  const twentyFourHoursAgo = new Date(Date.now() - 24 * 60 * 60 * 1000);

  return {
    ...state,
    recentCompletions: state.recentCompletions.filter((e) =>
      isRecentNotification(e, twentyFourHoursAgo),
    ),
    recentFailures: state.recentFailures.filter((e) =>
      isRecentNotification(e, twentyFourHoursAgo),
    ),
  };
}

export function calculateTotalCount(
  state: NotificationState,
): NotificationState {
  return {
    ...state,
    totalCount:
      state.activeExecutions.length +
      state.recentCompletions.length +
      state.recentFailures.length,
  };
}

export function handleExecutionUpdate(
  currentState: NotificationState,
  execution: GraphExecution,
  agentInfoMap: Map<
    string,
    { name: string; description: string; library_agent_id?: string }
  >,
): NotificationState {
  // Convert and enrich the execution
  const convertedExecution = convertLegacyExecutionToGenerated(execution);
  const enrichedExecution = enrichExecutionWithAgentInfo(
    convertedExecution,
    agentInfoMap,
  );

  // Remove from all categories first
  let newState = removeExecutionFromAllCategories(currentState, execution.id);

  // Add to appropriate category
  newState = addExecutionToCategory(newState, enrichedExecution);

  // Clean up old notifications
  newState = cleanupOldNotifications(newState);

  // Recalculate total count
  newState = calculateTotalCount(newState);

  return newState;
}
