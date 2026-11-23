import {
  getGetTaskQueryKey,
  getListTasksQueryKey,
  type getTaskResponse,
} from "@/app/api/__generated__/endpoints/tasks/tasks";
import type { Task } from "@/app/api/__generated__/models/task";
import type { QueryClient } from "@tanstack/react-query";

/** What a task list query holds: one page, or the pages of a longer list. */
function tasksIn(data: unknown): Task[] {
  if (!data || typeof data !== "object") return [];
  if ("pages" in data && Array.isArray(data.pages)) {
    return data.pages.flatMap(tasksIn);
  }
  if (!("status" in data) || data.status !== 200 || !("data" in data)) {
    return [];
  }
  const body = data.data;
  if (!body || typeof body !== "object" || !("tasks" in body)) return [];
  return Array.isArray(body.tasks) ? (body.tasks as Task[]) : [];
}

/** The task as a list already read it (the board, Home, the brief), so the
 *  panel can draw it while its own record loads. */
export function seedFromLists(
  queryClient: QueryClient,
  taskID: string,
): getTaskResponse | undefined {
  const cached = queryClient.getQueryData<getTaskResponse>(
    getGetTaskQueryKey(taskID),
  );
  if (cached?.status === 200) return cached;
  const lists = queryClient.getQueriesData({
    queryKey: getListTasksQueryKey(),
  });
  for (const [, data] of lists) {
    const task = tasksIn(data).find((item) => item.id === taskID);
    if (task) return { status: 200, data: task, headers: new Headers() };
  }
  return undefined;
}
