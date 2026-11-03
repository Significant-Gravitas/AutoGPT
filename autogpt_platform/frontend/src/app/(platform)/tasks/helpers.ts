import { PRIORITY_LABELS, STATUS_LABELS } from "./data/helpers";
import type {
  ExpertOption,
  InboxState,
  Task,
  TaskCounts,
  TaskFilters,
  TaskPriority,
  TaskStatus,
} from "./data/types";

export const TASK_VIEWS = ["list", "board"] as const;
export type TasksView = (typeof TASK_VIEWS)[number];

export const INBOX_TABS = [
  "all",
  "needs_you",
  "running",
  "waiting",
  "done",
  "failed",
] as const;
export type InboxTab = (typeof INBOX_TABS)[number];

export const INBOX_TAB_LABELS: Record<InboxTab, string> = {
  all: "All",
  needs_you: "Needs you",
  running: "Running",
  waiting: "Waiting",
  done: "Done",
  failed: "Failed",
};

export const BOARD_MIN_WIDTH = 620;

export function tabCount(counts: TaskCounts | undefined, tab: InboxTab) {
  return counts ? counts[tab] : 0;
}

export function matchesTab(task: Task, tab: InboxTab) {
  if (tab === "all") return true;
  if (tab === "needs_you") return task.inbox.needs_you;
  return task.inbox.state === tab;
}

const STATE_RANK: Record<InboxState, number> = {
  running: 1,
  waiting: 2,
  failed: 3,
  done: 4,
};

function inboxRank(task: Task) {
  return task.inbox.needs_you ? 0 : STATE_RANK[task.inbox.state];
}

export function sortForInbox(tasks: Task[]) {
  return [...tasks].sort((a, b) => inboxRank(a) - inboxRank(b));
}

export function filterInbox(tasks: Task[], tab: InboxTab) {
  return sortForInbox(tasks.filter((t) => matchesTab(t, tab)));
}

export type BoardColumnID = "running" | "waiting" | "done" | "failed";

export interface InboxColumn {
  id: BoardColumnID;
  label: string;
  tasks: Task[];
}

const COLUMN_LABELS: Record<BoardColumnID, string> = {
  running: "Running",
  waiting: "Waiting",
  done: "Done",
  failed: "Failed",
};

function boardColumnFor(task: Task, splitFailed: boolean): BoardColumnID {
  const state = task.inbox.state;
  if (task.inbox.needs_you) return "waiting";
  if (state === "failed") return splitFailed ? "failed" : "waiting";
  return state;
}

export function buildBoardColumns(tasks: Task[], tab: InboxTab) {
  const splitFailed = tab === "failed";
  const ids: BoardColumnID[] = splitFailed
    ? ["running", "waiting", "done", "failed"]
    : ["running", "waiting", "done"];
  const sorted = sortForInbox(tasks);
  return ids.map(
    (id): InboxColumn => ({
      id,
      label: COLUMN_LABELS[id],
      tasks: sorted.filter((t) => boardColumnFor(t, splitFailed) === id),
    }),
  );
}

export function inboxRowID(taskID: string) {
  return `inbox-row-${taskID}`;
}

export function describeLiveCounts(counts: TaskCounts | undefined) {
  if (!counts) return "";
  return `${counts.needs_you} need you, ${counts.running} running`;
}

export type DueFilter = "overdue" | "today" | "week";

export interface PageFilters {
  status: TaskStatus[];
  priority: TaskPriority[];
  assigneeIDs: string[];
  due: DueFilter | null;
}

export const EMPTY_FILTERS: PageFilters = {
  status: [],
  priority: [],
  assigneeIDs: [],
  due: null,
};

export const ALL_STATUSES: TaskStatus[] = [
  "backlog",
  "todo",
  "in_progress",
  "blocked",
  "in_review",
  "done",
  "cancelled",
];

export const ALL_PRIORITIES: TaskPriority[] = [
  "urgent",
  "high",
  "medium",
  "low",
  "none",
];

export const DUE_LABELS: Record<DueFilter, string> = {
  overdue: "Overdue",
  today: "Due today",
  week: "Due this week",
};

export function buildListFilters(args: {
  filters: PageFilters;
  mine: boolean;
}): TaskFilters {
  const { filters, mine } = args;
  return {
    status: filters.status.length ? filters.status : undefined,
    priority: filters.priority.length ? filters.priority : undefined,
    assignee_ids: filters.assigneeIDs.length ? filters.assigneeIDs : undefined,
    due: filters.due,
    mine: mine ? true : undefined,
  };
}

export interface FilterChip {
  id: string;
  label: string;
  next: PageFilters;
}

export function describeFilters(
  filters: PageFilters,
  experts: ExpertOption[],
): FilterChip[] {
  const chips: FilterChip[] = [];
  for (const s of filters.status) {
    chips.push({
      id: `status-${s}`,
      label: `Status: ${STATUS_LABELS[s]}`,
      next: { ...filters, status: filters.status.filter((x) => x !== s) },
    });
  }
  for (const p of filters.priority) {
    chips.push({
      id: `priority-${p}`,
      label: `Priority: ${PRIORITY_LABELS[p]}`,
      next: { ...filters, priority: filters.priority.filter((x) => x !== p) },
    });
  }
  for (const id of filters.assigneeIDs) {
    const name = experts.find((e) => e.id === id)?.name ?? "Expert";
    chips.push({
      id: `assignee-${id}`,
      label: `Expert: ${name}`,
      next: {
        ...filters,
        assigneeIDs: filters.assigneeIDs.filter((x) => x !== id),
      },
    });
  }
  if (filters.due) {
    chips.push({
      id: "due",
      label: DUE_LABELS[filters.due],
      next: { ...filters, due: null },
    });
  }
  return chips;
}

export function toggleValue<T>(list: T[], value: T) {
  return list.includes(value)
    ? list.filter((x) => x !== value)
    : [...list, value];
}

export function isTypingShortcutTarget(el: Element | null) {
  if (!el) return false;
  return !!el.closest("[role='dialog'], [role='menu'], [role='listbox']");
}
