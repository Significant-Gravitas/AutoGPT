import {
  getGetTaskQueryKey,
  getGetTaskRecordQueryKey,
  getListTasksQueryKey,
  type getTaskRecordResponse,
  type getTaskResponse,
  type listTasksResponse,
} from "@/app/api/__generated__/endpoints/tasks/tasks";
import type { Task } from "@/app/api/__generated__/models/task";
import type {
  InfiniteData,
  QueryClient,
  QueryKey,
} from "@tanstack/react-query";

/** A task as it will be once the change lands; null when it leaves the
 *  lists (archived). */
export type TaskPatch = (task: Task) => Task | null;

/** What the caches held before an optimistic change, to put back. */
export type CacheSnapshot = [QueryKey, unknown][];

function keysFor(taskID: string) {
  return [
    getListTasksQueryKey(),
    getGetTaskQueryKey(taskID),
    getGetTaskRecordQueryKey(taskID),
  ];
}

async function hold(queryClient: QueryClient, keys: QueryKey[]) {
  await Promise.all(
    keys.map((queryKey) => queryClient.cancelQueries({ queryKey })),
  );
  return keys.flatMap((queryKey) => queryClient.getQueriesData({ queryKey }));
}

/** A list read once, or page by page (the board). */
type ListCache = listTasksResponse | InfiniteData<listTasksResponse>;

function changePage(
  page: listTasksResponse,
  change: (tasks: Task[]) => Task[],
): listTasksResponse {
  return page.status === 200
    ? { ...page, data: { ...page.data, tasks: change(page.data.tasks) } }
    : page;
}

/** Change the tasks in every cached list, every page of a paged one, or
 *  only its first page for a task that belongs at the top. */
function patchLists(
  queryClient: QueryClient,
  change: (tasks: Task[]) => Task[],
  firstPageOnly = false,
) {
  queryClient.setQueriesData<ListCache>(
    { queryKey: getListTasksQueryKey() },
    (old) => {
      if (!old) return old;
      if (!("pages" in old)) return changePage(old, change);
      return {
        ...old,
        pages: old.pages.map((page, index) =>
          firstPageOnly && index > 0 ? page : changePage(page, change),
        ),
      };
    },
  );
}

/** Show a change to one task at once, in every list and in the task's own
 *  caches; returns what to put back if the server says no. */
export async function patchTask(
  queryClient: QueryClient,
  taskID: string,
  patch: TaskPatch,
): Promise<CacheSnapshot> {
  const [, taskKey, recordKey] = keysFor(taskID);
  const snapshot = await hold(queryClient, keysFor(taskID));
  patchLists(queryClient, (tasks) =>
    tasks.flatMap((task) => {
      if (task.id !== taskID) return [task];
      const next = patch(task);
      return next ? [next] : [];
    }),
  );
  queryClient.setQueryData<getTaskResponse>(taskKey, (old) => {
    const next = old?.status === 200 ? patch(old.data) : null;
    return old?.status === 200 && next ? { ...old, data: next } : old;
  });
  queryClient.setQueryData<getTaskRecordResponse>(recordKey, (old) => {
    const next = old?.status === 200 ? patch(old.data.task) : null;
    return old?.status === 200 && next
      ? { ...old, data: { ...old.data, task: next } }
      : old;
  });
  return snapshot;
}

/** Put a new task at the top of every list. */
export async function addTask(
  queryClient: QueryClient,
  task: Task,
): Promise<CacheSnapshot> {
  const snapshot = await hold(queryClient, [getListTasksQueryKey()]);
  patchLists(queryClient, (tasks) => [task, ...tasks], true);
  return snapshot;
}

/** Swap a stand-in for the task the server made. */
export function replaceTask(
  queryClient: QueryClient,
  standInID: string,
  task: Task,
) {
  patchLists(queryClient, (tasks) =>
    tasks.map((entry) => (entry.id === standInID ? task : entry)),
  );
}

/** The task as this browser last read it: its own read, else any list's. */
export function cachedTask(
  queryClient: QueryClient,
  taskID: string,
): Task | null {
  const own = queryClient.getQueryData<getTaskResponse>(
    getGetTaskQueryKey(taskID),
  );
  if (own?.status === 200) return own.data;
  for (const [, data] of queryClient.getQueriesData<ListCache>({
    queryKey: getListTasksQueryKey(),
  })) {
    const pages = data ? ("pages" in data ? data.pages : [data]) : [];
    for (const page of pages) {
      const found =
        page.status === 200
          ? page.data.tasks.find((task) => task.id === taskID)
          : undefined;
      if (found) return found;
    }
  }
  return null;
}

export function restore(queryClient: QueryClient, snapshot: CacheSnapshot) {
  for (const [queryKey, data] of snapshot) {
    queryClient.setQueryData(queryKey, data);
  }
}
