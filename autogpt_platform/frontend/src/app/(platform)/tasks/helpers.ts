import { PRIORITY_LABELS, STATUS_LABELS } from "./data/helpers";
import type {
  ExpertOption,
  Task,
  TaskFilters,
  TaskPriority,
  TaskStatus,
} from "./data/types";

export const TASK_VIEWS = ["board", "list", "mine"] as const;
export type TasksView = (typeof TASK_VIEWS)[number];

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

export const VISIBLE_STATUSES: TaskStatus[] = [
  "todo",
  "in_progress",
  "blocked",
  "in_review",
  "done",
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
  showHidden: boolean;
  view: TasksView;
}): TaskFilters {
  const { filters, showHidden, view } = args;
  const status = filters.status.length
    ? filters.status
    : showHidden
      ? ALL_STATUSES
      : undefined;
  return {
    status,
    priority: filters.priority.length ? filters.priority : undefined,
    assignee_ids: filters.assigneeIDs.length ? filters.assigneeIDs : undefined,
    due: filters.due,
    mine: view === "mine" ? true : undefined,
  };
}

export function groupByStatuses<
  T extends { id: string; statuses: TaskStatus[] },
>(tasks: Task[], columns: T[]) {
  return columns.map((column) => ({
    ...column,
    tasks: tasks.filter((t) => column.statuses.includes(t.status)),
  }));
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
