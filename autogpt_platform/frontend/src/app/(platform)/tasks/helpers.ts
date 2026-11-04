import {
  BUCKET_LABELS,
  bucketOf,
  PRIORITY_LABELS,
  statesIn,
  STATUS_BUCKETS,
  OTTO_FILTER_ID,
  type StatusBucket,
} from "./data/helpers";
import type {
  ExpertOption,
  Task,
  TaskCounts,
  TaskFilters,
  TaskPriority,
  TaskState,
} from "./data/types";

export const TASK_VIEWS = ["list", "board"] as const;
export type TasksView = (typeof TASK_VIEWS)[number];

export const LANES = STATUS_BUCKETS;
export type Lane = StatusBucket;

export const INBOX_TABS = ["all", ...LANES] as const;
export type InboxTab = (typeof INBOX_TABS)[number];

export const INBOX_TAB_LABELS: Record<InboxTab, string> = {
  all: "All",
  ...BUCKET_LABELS,
};

export function laneOf(task: Task) {
  return bucketOf(task);
}

export function tabCount(counts: TaskCounts | undefined, tab: InboxTab) {
  if (!counts) return 0;
  if (tab === "all") return counts.all;
  return statesIn(tab).reduce((n, s) => n + (counts.by_state[s] ?? 0), 0);
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
  mine: boolean;
  priority: TaskPriority[];
  assigneeIDs: string[];
  due: DueFilter | null;
}

export const EMPTY_FILTERS: PageFilters = {
  mine: false,
  priority: [],
  assigneeIDs: [],
  due: null,
};

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

export function hasFilters(filters: PageFilters) {
  return (
    filters.mine ||
    filters.priority.length > 0 ||
    filters.assigneeIDs.length > 0 ||
    filters.due !== null
  );
}

export function buildListFilters(filters: PageFilters): TaskFilters {
  return {
    mine: filters.mine || undefined,
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
  if (filters.mine) {
    chips.push({
      id: "mine",
      label: "Mine",
      next: { ...filters, mine: false },
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
    const name =
      id === OTTO_FILTER_ID
        ? "Otto"
        : (experts.find((e) => e.id === id)?.name ?? "Expert");
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

/** The Expert filter's choices: Otto first, then the team. */
export function expertChoices(experts: ExpertOption[]) {
  return [
    { id: OTTO_FILTER_ID, name: "Otto" },
    ...experts.map((e) => ({ id: e.id, name: e.name })),
  ];
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
