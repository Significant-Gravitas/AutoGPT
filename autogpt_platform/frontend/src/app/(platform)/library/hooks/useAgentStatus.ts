"use client";

import { useState } from "react";
import type {
  AgentStatus,
  AgentHealth,
  AgentStatusInfo,
  FleetSummary,
} from "../types";

/**
 * Derive health from status and recency.
 * TODO: Replace with real computation once backend provides the data.
 */
function deriveHealth(
  status: AgentStatus,
  lastRunAt: string | null,
): AgentHealth {
  if (status === "error") return "attention";
  if (status === "idle" && lastRunAt) {
    const daysSince =
      (Date.now() - new Date(lastRunAt).getTime()) / (1000 * 60 * 60 * 24);
    if (daysSince > 14) return "stale";
  }
  return "good";
}

/**
 * Generate deterministic mock status for an agent based on its ID.
 * This allows the UI to render realistic data before the real API is built.
 * TODO: Replace with real API call `GET /agents/:id/status`.
 */
function mockStatusForAgent(agentID: string): AgentStatusInfo {
  const hash = simpleHash(agentID);
  const statuses: AgentStatus[] = [
    "running",
    "error",
    "listening",
    "scheduled",
    "idle",
  ];
  const status = statuses[hash % statuses.length];
  const progress = status === "running" ? (hash * 17) % 100 : null;
  const totalRuns = (hash % 200) + 1;
  const daysAgo = (hash % 30) + 1;
  const lastRunAt = new Date(
    Date.now() - daysAgo * 24 * 60 * 60 * 1000,
  ).toISOString();
  const lastError =
    status === "error" ? "API rate limit exceeded — paused" : null;
  const monthlySpend = Number(((hash % 5000) / 100).toFixed(2));

  return {
    status,
    health: deriveHealth(status, lastRunAt),
    progress,
    totalRuns,
    lastRunAt,
    lastError,
    monthlySpend,
    nextScheduledRun:
      status === "scheduled"
        ? new Date(Date.now() + 3600_000).toISOString()
        : null,
    triggerType: status === "listening" ? "webhook" : null,
  };
}

function simpleHash(str: string): number {
  let h = 0;
  for (let i = 0; i < str.length; i++) {
    h = (h * 31 + str.charCodeAt(i)) >>> 0;
  }
  return h;
}

/**
 * Hook returning status info for a single agent.
 * TODO: Wire to `GET /agents/:id/status` + WebSocket `/agents/live`.
 */
export function useAgentStatus(agentID: string): AgentStatusInfo {
  // NOTE: useState initializer runs once on mount; a new agentID prop will not
  // re-derive info. Replace with a real API call wired to the agentID param.
  const [info] = useState(() => mockStatusForAgent(agentID));
  return info;
}

/**
 * Hook returning fleet-wide summary counts.
 * TODO: Wire to `GET /agents/summary`.
 */
export function useFleetSummary(agentIDs: string[]): FleetSummary {
  // NOTE: useState initializer runs once on mount; changes to agentIDs after
  // mount do NOT recompute the summary. Replace with a real API call wired to
  // the agentIDs prop once the backend endpoint is available.
  const [summary] = useState<FleetSummary>(() => {
    const counts: FleetSummary = {
      running: 0,
      error: 0,
      listening: 0,
      scheduled: 0,
      idle: 0,
      monthlySpend: 0,
    };
    for (const id of agentIDs) {
      const info = mockStatusForAgent(id);
      counts[info.status] += 1;
      counts.monthlySpend += info.monthlySpend;
    }
    counts.monthlySpend = Number(counts.monthlySpend.toFixed(2));
    return counts;
  });
  return summary;
}

export { mockStatusForAgent, deriveHealth };
