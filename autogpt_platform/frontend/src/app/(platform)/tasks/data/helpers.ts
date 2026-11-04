import type {
  Task,
  TaskMetrics,
  TaskPriority,
  TaskState,
  TaskStatus,
} from "./types";

export const BOARD_COLUMNS: {
  id: string;
  label: string;
  statuses: TaskStatus[];
}[] = [
  { id: "todo", label: "Todo", statuses: ["todo"] },
  {
    id: "in_progress",
    label: "In progress",
    statuses: ["in_progress", "in_review"],
  },
  { id: "needs_you", label: "Needs you", statuses: ["blocked"] },
  { id: "done", label: "Done", statuses: ["done"] },
];

export const HIDDEN_COLUMNS: {
  id: string;
  label: string;
  statuses: TaskStatus[];
}[] = [
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

export const STATE_LABELS: Record<TaskState, string> = {
  working: "Working",
  needs_you: "Needs you",
  stuck: "Stuck",
  done: "Done",
  stopped: "Stopped",
};

const ACTION_LABELS: Record<string, string> = {
  retry_failed_part: "Retry the failed part",
  skip: "Skip it",
  allow: "Allow",
  do_in_chat: "Do it in chat",
  change_goal: "Change goal",
  take_over: "Take over",
  stop: "Stop",
  accept: "Accept",
};

/** A worker can suggest an action of its own; it reads as its words. */
export function actionLabel(action: string) {
  const known = ACTION_LABELS[action];
  if (known) return known;
  const words = action.replace(/[_-]+/g, " ").trim();
  return words.charAt(0).toUpperCase() + words.slice(1);
}

/** Actions that need something from the user first, so they open the panel. */
export const ACTIONS_WITH_INPUT = new Set(["change_goal"]);

export type CardAction =
  | { kind: "answer"; label: string; askId: string }
  | { kind: "action"; label: string; action: string };

/** The one main button a card shows: the ask's default, the blocker's first
 *  way out, or accepting work that's waiting for an OK. */
export function cardAction(task: Task): CardAction | null {
  if (task.state === "done" || task.state === "stopped") return null;
  if (task.ask) {
    const pick = task.ask.default_option ?? task.ask.options?.[0];
    return pick ? { kind: "answer", label: pick, askId: task.ask.id } : null;
  }
  if (task.blocker) {
    const first = task.blocker.next_actions?.find(
      (a) => !ACTIONS_WITH_INPUT.has(a),
    );
    return first
      ? { kind: "action", label: actionLabel(first), action: first }
      : null;
  }
  if (task.raw_status === "awaiting_accept") {
    return { kind: "action", label: actionLabel("accept"), action: "accept" };
  }
  return null;
}

/** "now", "4m", "2h", "3d": how long ago, short enough for a card line. */
export function shortAgo(iso: string, now = Date.now()) {
  const minutes = Math.floor((now - new Date(iso).getTime()) / 60_000);
  if (minutes < 1) return "now";
  if (minutes < 60) return `${minutes}m`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h`;
  return `${Math.floor(hours / 24)}d`;
}

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
