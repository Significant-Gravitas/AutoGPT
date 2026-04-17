import type { Icon } from "@phosphor-icons/react";

export interface LibraryTab {
  id: string;
  title: string;
  icon: Icon;
}

/** Agent execution status — drives StatusBadge visuals & filtering. */
export type AgentStatus =
  | "running"
  | "error"
  | "listening"
  | "scheduled"
  | "idle";

/** Derived health bucket for quick triage. */
export type AgentHealth = "good" | "attention" | "stale";

/** Real-time metadata that powers the Intelligence Layer features. */
export interface AgentStatusInfo {
  status: AgentStatus;
  health: AgentHealth;
  /** 0-100 progress for currently running agents. */
  progress: number | null;
  totalRuns: number;
  lastRunAt: string | null;
  lastError: string | null;
  /** ID of the currently active execution (when status is "running"). */
  activeExecutionID: string | null;
  monthlySpend: number;
  nextScheduledRun: string | null;
  triggerType: string | null;
}

/** Fleet-wide aggregate counts used by the Briefing Panel stats grid. */
export interface FleetSummary {
  running: number;
  error: number;
  completed: number;
  listening: number;
  scheduled: number;
  idle: number;
  monthlySpend: number;
}

export type SitrepPriority =
  | "error"
  | "running"
  | "stale"
  | "success"
  | "listening"
  | "scheduled"
  | "idle";

export interface SitrepItemData {
  id: string;
  agentID: string;
  agentName: string;
  agentImageUrl?: string | null;
  executionID?: string;
  priority: SitrepPriority;
  message: string;
  status: AgentStatus;
}

/** Filter options for the agent filter dropdown. */
export type AgentStatusFilter =
  | "all"
  | "running"
  | "attention"
  | "completed"
  | "listening"
  | "scheduled"
  | "idle"
  | "healthy";
