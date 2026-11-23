import {
  getListTasksQueryKey,
  type listTasksResponse,
} from "@/app/api/__generated__/endpoints/tasks/tasks";
import type { ListTasksParams } from "@/app/api/__generated__/models/listTasksParams";
import type { Task } from "@/app/api/__generated__/models/task";
import type { InfiniteData } from "@tanstack/react-query";

/** Marks the paged list's cache entry apart from a single-page read with
 *  the same filters; it still starts with the list's key, so every change
 *  that re-reads or patches the lists reaches it. */
const PAGED = "pages";

export function pagedTasksKey(params: ListTasksParams) {
  return [...getListTasksQueryKey(params), PAGED] as const;
}

export function nextOffset(page: listTasksResponse) {
  if (page.status !== 200 || !page.data.has_more) return undefined;
  return page.data.next_offset ?? undefined;
}

/** Every task read so far, in order, each once: a task that moved up
 *  between two page reads would otherwise show twice. */
export function tasksOf(data: InfiniteData<listTasksResponse> | undefined) {
  const seen = new Set<string>();
  const tasks: Task[] = [];
  for (const page of data?.pages ?? []) {
    if (page.status !== 200) continue;
    for (const task of page.data.tasks) {
      if (seen.has(task.id)) continue;
      seen.add(task.id);
      tasks.push(task);
    }
  }
  return tasks;
}
