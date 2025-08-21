import { AgentExecutionStatus } from "@/app/api/__generated__/models/agentExecutionStatus";
import { GraphExecutionMeta as GeneratedGraphExecutionMeta } from "@/app/api/__generated__/models/graphExecutionMeta";
import { MyAgent } from "@/app/api/__generated__/models/myAgent";
import type { GraphExecution } from "@/lib/autogpt-server-api/types";

// Time constants
const MILLISECONDS_PER_SECOND = 1000;
const SECONDS_PER_MINUTE = 60;
const MINUTES_PER_HOUR = 60;
const MILLISECONDS_PER_MINUTE = SECONDS_PER_MINUTE * MILLISECONDS_PER_SECOND;
const MILLISECONDS_PER_HOUR = MINUTES_PER_HOUR * MILLISECONDS_PER_MINUTE;
const MILLISECONDS_PER_72_HOURS = 72 * MILLISECONDS_PER_HOUR;
const SHORT_DURATION_THRESHOLD_SECONDS = 5;
const MAX_EXECUTIONS_CAP = 1000;

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
  cutoffDate: Date,
): boolean {
  const status = execution.status;
  return (
    status === AgentExecutionStatus.COMPLETED &&
    !!execution.ended_at &&
    new Date(execution.ended_at) > cutoffDate
  );
}

export function isRecentFailure(
  execution: GeneratedGraphExecutionMeta,
  cutoffDate: Date,
): boolean {
  const status = execution.status;
  return (
    (status === AgentExecutionStatus.FAILED ||
      status === AgentExecutionStatus.TERMINATED) &&
    !!execution.ended_at &&
    new Date(execution.ended_at) > cutoffDate
  );
}

export function isRecentNotification(
  execution: AgentExecutionWithInfo,
  cutoffDate: Date,
): boolean {
  return execution.ended_at ? new Date(execution.ended_at) > cutoffDate : false;
}

export function categorizeExecutions(
  executions: GeneratedGraphExecutionMeta[],
  agentInfoMap: Map<
    string,
    { name: string; description: string; library_agent_id?: string }
  >,
): NotificationState {
  const seventyTwoHoursAgo = new Date(Date.now() - MILLISECONDS_PER_72_HOURS);

  // First filter by 72-hour window, then apply cap
  const recentExecutions = executions
    .filter((execution) => {
      // Always include active executions regardless of time
      if (isActiveExecution(execution)) return true;

      // For completed/failed executions, check if they're within 72 hours
      return (
        execution.ended_at && new Date(execution.ended_at) > seventyTwoHoursAgo
      );
    })
    .slice(0, MAX_EXECUTIONS_CAP); // Apply 800 execution cap

  const enrichedExecutions = recentExecutions.map((execution) =>
    enrichExecutionWithAgentInfo(execution, agentInfoMap),
  );

  // Categorize the capped executions
  const activeExecutions = enrichedExecutions.filter(isActiveExecution);

  const recentCompletions = enrichedExecutions.filter((execution) =>
    isRecentCompletion(execution, seventyTwoHoursAgo),
  );

  const recentFailures = enrichedExecutions.filter((execution) =>
    isRecentFailure(execution, seventyTwoHoursAgo),
  );

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
  const seventyTwoHoursAgo = new Date(Date.now() - MILLISECONDS_PER_72_HOURS);
  const newState = { ...state };

  if (isActiveExecution(execution)) {
    newState.activeExecutions = [execution, ...newState.activeExecutions];
  } else if (isRecentCompletion(execution, seventyTwoHoursAgo)) {
    newState.recentCompletions = [execution, ...newState.recentCompletions];
  } else if (isRecentFailure(execution, seventyTwoHoursAgo)) {
    newState.recentFailures = [execution, ...newState.recentFailures];
  }

  return newState;
}

export function cleanupOldNotifications(
  state: NotificationState,
): NotificationState {
  const seventyTwoHoursAgo = new Date(Date.now() - MILLISECONDS_PER_72_HOURS);

  const filteredRecentCompletions = state.recentCompletions.filter((e) =>
    isRecentNotification(e, seventyTwoHoursAgo),
  );
  const filteredRecentFailures = state.recentFailures.filter((e) =>
    isRecentNotification(e, seventyTwoHoursAgo),
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
