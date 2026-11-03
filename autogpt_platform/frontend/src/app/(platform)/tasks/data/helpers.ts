import type { TaskMetrics, TaskPriority, TaskStatus } from "./types";

export const BOARD_COLUMNS: { id: string; label: string; statuses: TaskStatus[] }[] = [
  { id: "todo", label: "Todo", statuses: ["todo"] },
  { id: "in_progress", label: "In progress", statuses: ["in_progress", "in_review"] },
  { id: "needs_you", label: "Needs you", statuses: ["blocked"] },
  { id: "done", label: "Done", statuses: ["done"] },
];

export const HIDDEN_COLUMNS: { id: string; label: string; statuses: TaskStatus[] }[] = [
  { id: "backlog", label: "Backlog", statuses: ["backlog"] },
  { id: "cancelled", label: "Cancelled", statuses: ["cancelled"] },
];

export const STATUS_LABELS: Record<TaskStatus, string> = {
  backlog: "Backlog",
  todo: "Todo",
  in_progress: "In progress",
  blocked: "Needs you",
  in_review: "In review",
  done: "Done",
  cancelled: "Cancelled",
};

export const PRIORITY_LABELS: Record<TaskPriority, string> = {
  none: "No priority",
  low: "Low",
  medium: "Medium",
  high: "High",
  urgent: "Urgent",
};

export function taskKey(number: number) {
  return `T-${number}`;
}

export function formatTokens(tokens: number) {
  if (tokens >= 1000) return `${Math.round(tokens / 1000)}k tokens`;
  return `${tokens} tokens`;
}

export function formatCost(usd: number) {
  return `$${usd.toFixed(2)}`;
}

export function formatDuration(ms: number) {
  const totalSeconds = Math.round(ms / 1000);
  const m = Math.floor(totalSeconds / 60);
  const s = totalSeconds % 60;
  if (m === 0) return `${s}s`;
  return s ? `${m}m ${s}s` : `${m}m`;
}

/** "4 tool calls · 12k tokens · $0.04 · 1m 20s" */
export function formatMetrics(m: TaskMetrics) {
  return [
    `${m.tool_calls} tool call${m.tool_calls === 1 ? "" : "s"}`,
    formatTokens(m.tokens),
    formatCost(m.cost_usd),
    formatDuration(m.duration_ms),
  ].join(" · ");
}
