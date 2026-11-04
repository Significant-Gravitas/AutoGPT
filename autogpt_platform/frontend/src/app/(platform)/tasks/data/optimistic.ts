import type { QueryClient, QueryKey } from "@tanstack/react-query";
import type {
  ExpertOption,
  Task,
  TaskAssignee,
  TaskCounts,
  UpdateTaskInput,
} from "./types";

// Optimistic edits for the task caches: every list, the inbox (tasks plus
// counts), Needs you, a task's detail and its summary.

export type CacheEdit =
  | { kind: "update"; changes: Partial<Task> }
  | { kind: "remove" };

type Snapshot = [QueryKey, unknown][];

const TASK_CACHES = new Set(["list", "needs-you", "detail", "summary"]);

type Row = { id?: unknown; state?: Task["state"] };

function isRow(value: unknown): value is Row {
  return !!value && typeof value === "object" && "id" in value;
}

function patchRows(rows: unknown[], id: string, edit: CacheEdit) {
  if (edit.kind === "remove") {
    return rows.filter((row) => !isRow(row) || row.id !== id);
  }
  return rows.map((row) =>
    isRow(row) && row.id === id ? { ...row, ...edit.changes } : row,
  );
}

function dropFromCounts(counts: TaskCounts, removed: Row | undefined) {
  if (!removed?.state) return counts;
  const state = removed.state;
  return {
    ...counts,
    all: Math.max(0, counts.all - 1),
    by_state: {
      ...counts.by_state,
      [state]: Math.max(0, (counts.by_state[state] ?? 0) - 1),
    },
  };
}

function patchInbox(
  data: { tasks: unknown[]; counts?: TaskCounts },
  id: string,
  edit: CacheEdit,
) {
  const tasks = patchRows(data.tasks, id, edit);
  if (edit.kind !== "remove" || !data.counts) return { ...data, tasks };
  const removed = data.tasks.find(
    (row): row is Row => isRow(row) && row.id === id,
  );
  return { ...data, tasks, counts: dropFromCounts(data.counts, removed) };
}

function patchData(data: unknown, id: string, edit: CacheEdit): unknown {
  if (Array.isArray(data)) return patchRows(data, id, edit);
  if (!data || typeof data !== "object") return data;
  if ("tasks" in data && Array.isArray((data as { tasks: unknown }).tasks)) {
    return patchInbox(data as { tasks: unknown[] }, id, edit);
  }
  // A detail or summary: a removed task keeps its copy until the panel closes.
  if (isRow(data) && data.id === id && edit.kind === "update") {
    return { ...data, ...edit.changes };
  }
  return data;
}

function isTaskCache(queryKey: QueryKey) {
  return typeof queryKey[1] === "string" && TASK_CACHES.has(queryKey[1]);
}

export function patchTaskCaches(
  queryClient: QueryClient,
  id: string,
  edit: CacheEdit,
): Snapshot {
  const filter = {
    queryKey: ["tasks-v1"],
    predicate: (q: { queryKey: QueryKey }) => isTaskCache(q.queryKey),
  };
  const snapshot = queryClient.getQueriesData(filter);
  queryClient.setQueriesData(filter, (data: unknown) =>
    patchData(data, id, edit),
  );
  return snapshot;
}

export function restoreTaskCaches(
  queryClient: QueryClient,
  snapshot: Snapshot,
) {
  for (const [queryKey, data] of snapshot) {
    queryClient.setQueryData(queryKey, data);
  }
}

/** The assignee to show before the server answers a reassign. */
export function optimisticAssignee(
  input: NonNullable<UpdateTaskInput["assignee"]> | null,
  experts: ExpertOption[],
): TaskAssignee | null {
  if (!input) return null;
  if (input.type === "otto") return { type: "otto", id: null, name: "Otto" };
  if (input.type === "user") return { type: "user", id: input.id, name: "Me" };
  const expert = experts.find((e) => e.id === input.id);
  return {
    type: "expert",
    id: input.id,
    name: expert?.name ?? "Expert",
    avatar_url: expert?.avatar_url ?? null,
  };
}
