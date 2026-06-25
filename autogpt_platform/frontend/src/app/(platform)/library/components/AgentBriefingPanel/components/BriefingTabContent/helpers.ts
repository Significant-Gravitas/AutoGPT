import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import type { AgentStatusFilter } from "@/app/(platform)/library/types";

export const MAX_VISIBLE = 6;

const TAB_STATUS_LABEL: Record<string, string> = {
  listening: "Waiting for trigger event",
  scheduled: "Has a scheduled run",
  idle: "No recent activity",
};

export function getAgentStatusLabel(tab: string, agent: LibraryAgent): string {
  if (tab === "scheduled" && agent.next_scheduled_run) {
    const diff = new Date(agent.next_scheduled_run).getTime() - Date.now();
    const minutes = Math.round(diff / 60_000);
    if (minutes <= 0) return "Scheduled to run soon";
    if (minutes < 60) return `Scheduled to run in ${minutes}m`;
    const hours = Math.round(minutes / 60);
    if (hours < 24) return `Scheduled to run in ${hours}h`;
    const days = Math.round(hours / 24);
    return `Scheduled to run in ${days}d`;
  }
  return TAB_STATUS_LABEL[tab] ?? "";
}

const EMPTY_MESSAGES: Record<string, string> = {
  running: "No agents running right now",
  attention: "No agents that need attention",
  completed: "No recently completed tasks",
  listening: "No agents listening for events",
  scheduled: "No agents with scheduled tasks",
  idle: "No idle agents",
};

export function getEmptyMessage(tab: AgentStatusFilter): string {
  return EMPTY_MESSAGES[tab] ?? "No agents in this category";
}
