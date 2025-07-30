import { AgentExecutionStatus } from "@/app/api/__generated__/models/agentExecutionStatus";
import { GraphExecutionMeta as GeneratedGraphExecutionMeta } from "@/app/api/__generated__/models/graphExecutionMeta";
import { MyAgent } from "@/app/api/__generated__/models/myAgent";
import type { GraphExecution } from "@/lib/autogpt-server-api/types";

// Time constants
const MILLISECONDS_PER_SECOND = 1000;
const SECONDS_PER_MINUTE = 60;
const MINUTES_PER_HOUR = 60;
const HOURS_PER_DAY = 24;
const DAYS_PER_WEEK = 7;
const MILLISECONDS_PER_MINUTE = SECONDS_PER_MINUTE * MILLISECONDS_PER_SECOND;
const MILLISECONDS_PER_HOUR = MINUTES_PER_HOUR * MILLISECONDS_PER_MINUTE;
const MILLISECONDS_PER_DAY = HOURS_PER_DAY * MILLISECONDS_PER_HOUR;
const MILLISECONDS_PER_WEEK = DAYS_PER_WEEK * MILLISECONDS_PER_DAY;

// Display constants
const SHORT_DURATION_THRESHOLD_SECONDS = 5;

// State sanity limits - keep only most recent executions to prevent unbounded growth
const MAX_ACTIVE_EXECUTIONS_IN_STATE = 200; // Most important - these are running
const MAX_RECENT_COMPLETIONS_IN_STATE = 100;
const MAX_RECENT_FAILURES_IN_STATE = 100;

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
    // Ensure we have valid agent data
    const agentName =
      agent.agent_name || `Agent ${agent.agent_id?.slice(0, 8)}`;
    const agentDescription = agent.description || "";

    agentMap.set(agent.agent_id, {
      name: agentName,
      description: agentDescription,
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
    agent_name: agentInfo?.name ?? "Unknown Agent",
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
  oneWeekAgo: Date,
): boolean {
  const status = execution.status;
  return (
    status === AgentExecutionStatus.COMPLETED &&
    !!execution.ended_at &&
    new Date(execution.ended_at) > oneWeekAgo
  );
}

export function isRecentFailure(
  execution: GeneratedGraphExecutionMeta,
  oneWeekAgo: Date,
): boolean {
  const status = execution.status;
  return (
    (status === AgentExecutionStatus.FAILED ||
      status === AgentExecutionStatus.TERMINATED) &&
    !!execution.ended_at &&
    new Date(execution.ended_at) > oneWeekAgo
  );
}

export function isRecentNotification(
  execution: AgentExecutionWithInfo,
  oneWeekAgo: Date,
): boolean {
  return execution.ended_at ? new Date(execution.ended_at) > oneWeekAgo : false;
}

export function categorizeExecutions(
  executions: GeneratedGraphExecutionMeta[],
  agentInfoMap: Map<
    string,
    { name: string; description: string; library_agent_id?: string }
  >,
): NotificationState {
  const oneWeekAgo = new Date(Date.now() - MILLISECONDS_PER_WEEK);

  const enrichedExecutions = executions.map((execution) =>
    enrichExecutionWithAgentInfo(execution, agentInfoMap),
  );

  // Filter and limit each category to prevent unbounded state growth
  const activeExecutions = enrichedExecutions
    .filter(isActiveExecution)
    .slice(0, MAX_ACTIVE_EXECUTIONS_IN_STATE);

  const recentCompletions = enrichedExecutions
    .filter((execution) => isRecentCompletion(execution, oneWeekAgo))
    .slice(0, MAX_RECENT_COMPLETIONS_IN_STATE);

  const recentFailures = enrichedExecutions
    .filter((execution) => isRecentFailure(execution, oneWeekAgo))
    .slice(0, MAX_RECENT_FAILURES_IN_STATE);

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
  };
}

export function addExecutionToCategory(
  state: NotificationState,
  execution: AgentExecutionWithInfo,
): NotificationState {
  const oneWeekAgo = new Date(Date.now() - MILLISECONDS_PER_WEEK);
  const newState = { ...state };

  if (isActiveExecution(execution)) {
    newState.activeExecutions = [execution, ...newState.activeExecutions].slice(
      0,
      MAX_ACTIVE_EXECUTIONS_IN_STATE,
    );
  } else if (isRecentCompletion(execution, oneWeekAgo)) {
    newState.recentCompletions = [
      execution,
      ...newState.recentCompletions,
    ].slice(0, MAX_RECENT_COMPLETIONS_IN_STATE);
  } else if (isRecentFailure(execution, oneWeekAgo)) {
    newState.recentFailures = [execution, ...newState.recentFailures].slice(
      0,
      MAX_RECENT_FAILURES_IN_STATE,
    );
  }

  return newState;
}

export function cleanupOldNotifications(
  state: NotificationState,
): NotificationState {
  const oneWeekAgo = new Date(Date.now() - MILLISECONDS_PER_WEEK);

  const filteredRecentCompletions = state.recentCompletions.filter((e) =>
    isRecentNotification(e, oneWeekAgo),
  );
  const filteredRecentFailures = state.recentFailures.filter((e) =>
    isRecentNotification(e, oneWeekAgo),
  );

  return {
    ...state,
    recentCompletions: filteredRecentCompletions,
    recentFailures: filteredRecentFailures,
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
