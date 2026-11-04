import type {
  Task,
  TaskMetrics,
  TaskPriority,
  TaskState,
  TaskStatus,
} from "./types";

export const STATUS_LABELS: Record<TaskStatus, string> = {
  backlog: "Backlog",
  todo: "Todo",
  in_progress: "In progress",
  blocked: "Needs you",
  in_review: "In review",
  done: "Done",
  cancelled: "Cancelled",
};

/** The four buckets every surface sorts a task into: tabs, lanes, badges. */
export const STATUS_BUCKETS = [
  "needs_you",
  "working",
  "done",
  "cancelled",
] as const;
export type StatusBucket = (typeof STATUS_BUCKETS)[number];

export const BUCKET_LABELS: Record<StatusBucket, string> = {
  needs_you: "Needs you",
  working: "Working",
  done: "Done",
  cancelled: "Cancelled",
};

export type StatusTone = "waiting" | "error" | "info" | "ok" | "quiet";

export interface TaskStatusView {
  bucket: StatusBucket;
  /** The one word the badge shows. */
  word: string;
  tone: StatusTone;
  /** A failed task sits in Needs you, flagged red. */
  failed: boolean;
}

const STATUS_VIEWS: Record<TaskState, TaskStatusView> = {
  needs_you: {
    bucket: "needs_you",
    word: "Needs you",
    tone: "waiting",
    failed: false,
  },
  stuck: { bucket: "needs_you", word: "Failed", tone: "error", failed: true },
  working: { bucket: "working", word: "Working", tone: "info", failed: false },
  done: { bucket: "done", word: "Done", tone: "ok", failed: false },
  stopped: {
    bucket: "cancelled",
    word: "Cancelled",
    tone: "quiet",
    failed: false,
  },
};

/** The single status vocabulary: which bucket, which word, which colour. */
export function statusOf(task: TaskState | { state: TaskState }) {
  const state = typeof task === "string" ? task : task.state;
  return STATUS_VIEWS[state] ?? STATUS_VIEWS.working;
}

export function bucketOf(task: TaskState | { state: TaskState }) {
  return statusOf(task).bucket;
}

/** The states that fall in a bucket, for counting tabs and lanes. */
export function statesIn(bucket: StatusBucket) {
  return (Object.keys(STATUS_VIEWS) as TaskState[]).filter(
    (state) => STATUS_VIEWS[state].bucket === bucket,
  );
}

/** One short line under a task's title: its result when done, why it failed,
 *  what it needs from you, or the step it's on. Never a second status word. */
export function statusLine(task: Task): string | null {
  switch (task.state) {
    case "done":
      return task.outcome_summary || null;
    case "stuck":
      return (
        task.blocker?.short || task.blocker?.what_stopped_it || task.now || null
      );
    case "needs_you":
      if (task.approvals?.length) return null;
      return (
        task.ask?.question ||
        task.blocker?.short ||
        task.outcome_summary ||
        task.now ||
        null
      );
    case "working":
      return task.live?.current_step || task.now || null;
    default:
      return null;
  }
}

/** How long, worded for the bucket: "20s ago" for the live step, "waiting
 *  5h" for something on you, "2d ago" for finished work. */
export function statusAge(task: Task, now = Date.now()) {
  const bucket = bucketOf(task);
  const lastStep = task.inbox?.steps?.at(-1)?.time;
  const at =
    bucket === "working"
      ? (lastStep ?? task.updated_at)
      : bucket === "needs_you"
        ? task.updated_at
        : (task.completed_at ?? task.updated_at);
  const ms = now - Date.parse(at);
  if (!Number.isFinite(ms)) return null;
  const age = formatAge(ms);
  if (bucket === "needs_you" && !statusOf(task).failed) return `waiting ${age}`;
  return `${age} ago`;
}

/** Otto in the Expert filter: Otto's tasks carry no expert id. */
export const OTTO_FILTER_ID = "otto";

/** Whether a task belongs to one of the picked assignees (Otto included). */
export function matchesAssignees(task: Task, ids: string[]) {
  if (ids.includes(OTTO_FILTER_ID) && task.assignee?.type === "otto")
    return true;
  return !!task.assignee?.id && ids.includes(task.assignee.id);
}

export const PRIORITY_LABELS: Record<TaskPriority, string> = {
  none: "No priority",
  low: "Low",
  medium: "Medium",
  high: "High",
  urgent: "Urgent",
};

export const STATE_LABELS: Record<TaskState, string> = {
  working: statusOf("working").word,
  needs_you: statusOf("needs_you").word,
  stuck: statusOf("stuck").word,
  done: statusOf("done").word,
  stopped: statusOf("stopped").word,
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

/** The one duration formatter: "20s", "4m", "5h", "2d". */
export function formatAge(ms: number) {
  const seconds = Math.max(0, Math.floor(ms / 1000));
  if (seconds < 60) return `${seconds}s`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h`;
  return `${Math.floor(hours / 24)}d`;
}

/** "now", "4m", "2h", "3d": how long ago, short enough for a card line. */
export function shortAgo(iso: string, now = Date.now()) {
  const ms = now - new Date(iso).getTime();
  if (!Number.isFinite(ms) || ms < 60_000) return "now";
  return formatAge(ms);
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
  return formatAge(ms);
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
