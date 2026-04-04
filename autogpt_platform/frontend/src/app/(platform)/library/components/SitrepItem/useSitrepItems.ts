"use client";

import { useMemo } from "react";
import { mockStatusForAgent } from "../../hooks/useAgentStatus";
import type { SitrepItemData, SitrepPriority } from "./SitrepItem";
import type { AgentStatus } from "../../types";

/**
 * Produce a prioritised list of sitrep items from agent IDs.
 * Priority order: error → running → stale → success.
 *
 * TODO: Replace with `GET /agents/sitrep` once the backend endpoint exists.
 */
export function useSitrepItems(
  agentIDs: string[],
  maxItems: number,
): SitrepItemData[] {
  const items = useMemo<SitrepItemData[]>(() => {
    const raw: SitrepItemData[] = agentIDs.map((id) => {
      const info = mockStatusForAgent(id);
      return {
        id,
        agentID: id,
        agentName: `Agent ${id.slice(0, 6)}`,
        priority: toPriority(info.status, info.health === "stale"),
        message: buildMessage(info.status, info.lastError, info.progress),
        status: info.status,
      };
    });

    const order: Record<SitrepPriority, number> = {
      error: 0,
      running: 1,
      stale: 2,
      success: 3,
    };
    raw.sort((a, b) => order[a.priority] - order[b.priority]);

    return raw.slice(0, maxItems);
  }, [agentIDs, maxItems]);

  return items;
}

function toPriority(status: AgentStatus, isStale: boolean): SitrepPriority {
  if (status === "error") return "error";
  if (status === "running") return "running";
  if (isStale || status === "idle") return "stale";
  return "success";
}

function buildMessage(
  status: AgentStatus,
  lastError: string | null,
  progress: number | null,
): string {
  switch (status) {
    case "error":
      return lastError ?? "Unknown error occurred";
    case "running":
      return progress !== null
        ? `${progress}% complete`
        : "Currently executing";
    case "idle":
      return "Hasn't run recently — still relevant?";
    case "listening":
      return "Waiting for trigger event";
    case "scheduled":
      return "Next run scheduled";
  }
}
