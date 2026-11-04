import { PRIORITY_LABELS, STATUS_LABELS } from "./data/helpers";
import type {
  ExpertOption,
  Task,
  TaskCounts,
  TaskFilters,
  TaskPriority,
  TaskState,
  TaskStatus,
} from "./data/types";

export const TASK_VIEWS = ["list", "board"] as const;
export type TasksView = (typeof TASK_VIEWS)[number];

export const LANES = ["needs_you", "working", "done", "cancelled"] as const;
export type Lane = (typeof LANES)[number];

export const INBOX_TABS = ["all", ...LANES] as const;
export type InboxTab = (typeof INBOX_TABS)[number];

export const INBOX_TAB_LABELS: Record<InboxTab, string> = {
  all: "All",
  needs_you: "Needs you",
  working: "Working",
  done: "Done",
  cancelled: "Cancelled",
};

const LANE_OF: Record<TaskState, Lane> = {
  needs_you: "needs_you",
  stuck: "needs_you",
  working: "working",
  done: "done",
  stopped: "cancelled",
};

const LANE_STATES: Record<Lane, TaskState[]> = {
  needs_you: ["needs_you", "stuck"],
  working: ["working"],
  done: ["done"],
  cancelled: ["stopped"],
};

export const BOARD_MIN_WIDTH = 820;

export function laneOf(task: Task) {
  return LANE_OF[task.state];
}

export function tabCount(counts: TaskCounts | undefined, tab: InboxTab) {
  if (!counts) return 0;
  if (tab === "all") return counts.all;
  return LANE_STATES[tab].reduce((n, s) => n + (counts.by_state[s] ?? 0), 0);
}

const LANE_RANK: Record<Lane, number> = {
  needs_you: 0,
  working: 1,
  done: 2,
  cancelled: 3,
};

function byLaneThenRecent(a: Task, b: Task) {
  const lane = LANE_RANK[laneOf(a)] - LANE_RANK[laneOf(b)];
  if (lane) return lane;
  return b.updated_at.localeCompare(a.updated_at);
}

export function filterInbox(tasks: Task[], tab: InboxTab) {
  const shown = tab === "all" ? tasks : tasks.filter((t) => laneOf(t) === tab);
  return shown.toSorted(byLaneThenRecent);
}

export interface InboxColumn {
  id: Lane;
  label: string;
  tasks: Task[];
}

export function buildBoardColumns(tasks: Task[]) {
  return LANES.map(
    (id): InboxColumn => ({
      id,
      label: INBOX_TAB_LABELS[id],
      tasks: tasks.filter((t) => laneOf(t) === id),
    }),
  );
}

export function inboxRowID(taskID: string) {
  return `inbox-row-${taskID}`;
}

export function describeLiveCounts(counts: TaskCounts | undefined) {
  if (!counts) return "";
  return `${tabCount(counts, "needs_you")} need you, ${tabCount(counts, "working")} working`;
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

export function buildListFilters(filters: PageFilters): TaskFilters {
  return {
    status: filters.status.length ? filters.status : undefined,
    priority: filters.priority.length ? filters.priority : undefined,
    assignee_ids: filters.assigneeIDs.length ? filters.assigneeIDs : undefined,
    due: filters.due,
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

/** A task that is working or waiting on the user can still be cancelled. */
export function isCancellable(state: TaskState) {
  return state === "working" || state === "needs_you" || state === "stuck";
}
