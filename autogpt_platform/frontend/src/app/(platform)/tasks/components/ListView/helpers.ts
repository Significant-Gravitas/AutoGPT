import type { Task, TaskStatus } from "../../data/types";

export const LIST_GROUP_ORDER: TaskStatus[] = [
  "blocked",
  "in_progress",
  "in_review",
  "todo",
  "backlog",
  "done",
  "cancelled",
];

export function groupTasksByStatus(tasks: Task[]) {
  return LIST_GROUP_ORDER.map((status) => ({
    status,
    tasks: tasks.filter((t) => t.status === status),
  })).filter((group) => group.tasks.length > 0);
}

export function statusBadgeVariant(status: TaskStatus) {
  if (status === "blocked") return "warning" as const;
  if (status === "done") return "success" as const;
  if (status === "cancelled") return "error" as const;
  return "info" as const;
}
