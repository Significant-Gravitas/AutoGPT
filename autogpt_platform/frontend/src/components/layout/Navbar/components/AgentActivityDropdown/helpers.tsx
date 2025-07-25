import { AgentExecutionStatus } from "@/app/api/__generated__/models/agentExecutionStatus";
import { GraphExecutionMeta as GeneratedGraphExecutionMeta } from "@/app/api/__generated__/models/graphExecutionMeta";
import { MyAgent } from "@/app/api/__generated__/models/myAgent";
import type { GraphExecution } from "@/lib/autogpt-server-api/types";

// Time constants
const MILLISECONDS_PER_SECOND = 1000;
const SECONDS_PER_MINUTE = 60;
const MINUTES_PER_HOUR = 60;
const HOURS_PER_DAY = 24;
const MILLISECONDS_PER_MINUTE = SECONDS_PER_MINUTE * MILLISECONDS_PER_SECOND;
const MILLISECONDS_PER_HOUR = MINUTES_PER_HOUR * MILLISECONDS_PER_MINUTE;
const MILLISECONDS_PER_DAY = HOURS_PER_DAY * MILLISECONDS_PER_HOUR;

// Display constants
export const EXECUTION_DISPLAY_LIMIT = 6;
const SHORT_DURATION_THRESHOLD_SECONDS = 5;

export function formatTimeAgo(dateStr: string): string {
  const date = new Date(dateStr);
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffMins = Math.floor(diffMs / MILLISECONDS_PER_MINUTE);

  if (diffMins < 1) return "just now";
  if (diffMins < SECONDS_PER_MINUTE) return `${diffMins}m ago`;
  const diffHours = Math.floor(diffMins / MINUTES_PER_HOUR);
  if (diffHours < HOURS_PER_DAY) return `${diffHours}h ago`;
  const diffDays = Math.floor(diffHours / HOURS_PER_DAY);
  return `${diffDays}d ago`;
}

export function getExecutionDuration(
  execution: GeneratedGraphExecutionMeta,
): string {
  if (!execution.started_at) return "Unknown";

  const start = new Date(execution.started_at);
  const end = execution.ended_at ? new Date(execution.ended_at) : new Date();

  // Check if dates are valid
  if (isNaN(start.getTime()) || isNaN(end.getTime())) {
    return "Unknown";
  }

  const durationMs = end.getTime() - start.getTime();

  // Handle negative durations (shouldn't happen but just in case)
  if (durationMs < 0) return "Unknown";

  const durationSec = Math.floor(durationMs / MILLISECONDS_PER_SECOND);

  // For short durations (< 5 seconds), show "a few seconds"
  if (durationSec < SHORT_DURATION_THRESHOLD_SECONDS) {
    return "a few seconds";
  }

  if (durationSec < SECONDS_PER_MINUTE) return `${durationSec}s`;
  const durationMin = Math.floor(durationSec / SECONDS_PER_MINUTE);
  if (durationMin < MINUTES_PER_HOUR)
    return `${durationMin}m ${durationSec % SECONDS_PER_MINUTE}s`;
  const durationHr = Math.floor(durationMin / MINUTES_PER_HOUR);
  return `${durationHr}h ${durationMin % MINUTES_PER_HOUR}m`;
}

export function formatNotificationCount(count: number): string {
  if (count > 99) return "+99";
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
  activeCount: number;
  recentCompletionsCount: number;
  recentFailuresCount: number;
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
  const twentyFourHoursAgo = new Date(Date.now() - MILLISECONDS_PER_DAY);

  const enrichedExecutions = executions.map((execution) =>
    enrichExecutionWithAgentInfo(execution, agentInfoMap),
  );

  const allActiveExecutions = enrichedExecutions.filter(isActiveExecution);
  const allRecentCompletions = enrichedExecutions.filter((execution) =>
    isRecentCompletion(execution, twentyFourHoursAgo),
  );
  const allRecentFailures = enrichedExecutions.filter((execution) =>
    isRecentFailure(execution, twentyFourHoursAgo),
  );

  // Limited arrays for dropdown display
  const activeExecutions = allActiveExecutions.slice(
    0,
    EXECUTION_DISPLAY_LIMIT,
  );
  const recentCompletions = allRecentCompletions.slice(
    0,
    EXECUTION_DISPLAY_LIMIT,
  );
  const recentFailures = allRecentFailures.slice(0, EXECUTION_DISPLAY_LIMIT);

  // Counts for badge/hints
  const activeCount = allActiveExecutions.length;
  const recentCompletionsCount = allRecentCompletions.length;
  const recentFailuresCount = allRecentFailures.length;

  return {
    activeExecutions,
    recentCompletions,
    recentFailures,
    totalCount: activeCount + recentCompletionsCount + recentFailuresCount,
    activeCount,
    recentCompletionsCount,
    recentFailuresCount,
  };
}

export function removeExecutionFromAllCategories(
  state: NotificationState,
  executionId: string,
): NotificationState {
  const filteredActiveExecutions = state.activeExecutions.filter(
    (e) => e.id !== executionId,
  );
  const filteredRecentCompletions = state.recentCompletions.filter(
    (e) => e.id !== executionId,
  );
  const filteredRecentFailures = state.recentFailures.filter(
    (e) => e.id !== executionId,
  );

  return {
    activeExecutions: filteredActiveExecutions,
    recentCompletions: filteredRecentCompletions,
    recentFailures: filteredRecentFailures,
    totalCount: state.totalCount, // Will be recalculated later
    activeCount: Math.max(
      0,
      state.activeCount -
        (state.activeExecutions.length - filteredActiveExecutions.length),
    ),
    recentCompletionsCount: Math.max(
      0,
      state.recentCompletionsCount -
        (state.recentCompletions.length - filteredRecentCompletions.length),
    ),
    recentFailuresCount: Math.max(
      0,
      state.recentFailuresCount -
        (state.recentFailures.length - filteredRecentFailures.length),
    ),
  };
}

export function addExecutionToCategory(
  state: NotificationState,
  execution: AgentExecutionWithInfo,
): NotificationState {
  const twentyFourHoursAgo = new Date(Date.now() - MILLISECONDS_PER_DAY);
  const newState = { ...state };

  if (isActiveExecution(execution)) {
    newState.activeExecutions = [execution, ...newState.activeExecutions].slice(
      0,
      EXECUTION_DISPLAY_LIMIT,
    );
    newState.activeCount = newState.activeCount + 1;
  } else if (isRecentCompletion(execution, twentyFourHoursAgo)) {
    newState.recentCompletions = [
      execution,
      ...newState.recentCompletions,
    ].slice(0, EXECUTION_DISPLAY_LIMIT);
    newState.recentCompletionsCount = newState.recentCompletionsCount + 1;
  } else if (isRecentFailure(execution, twentyFourHoursAgo)) {
    newState.recentFailures = [execution, ...newState.recentFailures].slice(
      0,
      EXECUTION_DISPLAY_LIMIT,
    );
    newState.recentFailuresCount = newState.recentFailuresCount + 1;
  }

  return newState;
}

export function cleanupOldNotifications(
  state: NotificationState,
): NotificationState {
  const twentyFourHoursAgo = new Date(Date.now() - 24 * 60 * 60 * 1000);

  const filteredRecentCompletions = state.recentCompletions.filter((e) =>
    isRecentNotification(e, twentyFourHoursAgo),
  );
  const filteredRecentFailures = state.recentFailures.filter((e) =>
    isRecentNotification(e, twentyFourHoursAgo),
  );

  return {
    ...state,
    recentCompletions: filteredRecentCompletions,
    recentFailures: filteredRecentFailures,
    recentCompletionsCount: Math.max(
      0,
      state.recentCompletionsCount -
        (state.recentCompletions.length - filteredRecentCompletions.length),
    ),
    recentFailuresCount: Math.max(
      0,
      state.recentFailuresCount -
        (state.recentFailures.length - filteredRecentFailures.length),
    ),
  };
}

export function calculateTotalCount(
  state: NotificationState,
): NotificationState {
  return {
    ...state,
    totalCount:
      state.activeCount +
      state.recentCompletionsCount +
      state.recentFailuresCount,
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
