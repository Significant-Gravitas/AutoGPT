import { useListExperts } from "@/app/api/__generated__/endpoints/experts/experts";
import { listTasks } from "@/app/api/__generated__/endpoints/tasks/tasks";
import { okData } from "@/app/api/helpers";
import { pollInterval } from "@/app/(platform)/tasks/helpers";
import { useTaskSync } from "@/app/(platform)/tasks/taskSync";
import { keepPreviousData, useInfiniteQuery } from "@tanstack/react-query";
import { useWorkAreaStore } from "../../../../workArea";
import { nextOffset, pagedTasksKey, tasksOf } from "./pagedTasks";
import { listParams } from "./taskFilters";
import { useTaskFilters } from "./useTaskFilters";

/** A tab's name until its task is read. */
const UNREAD_TITLE = "Task";

export function useHomeBoard() {
  const openTab = useWorkAreaStore((s) => s.openTask);
  const openTabs = useWorkAreaStore((s) => s.tabs);
  const activeKey = useWorkAreaStore((s) => s.activeKey);
  const isFolded = useWorkAreaStore((s) => s.isFolded);
  const filters = useTaskFilters();
  const params = listParams(filters.applied);
  const tasksQuery = useInfiniteQuery({
    queryKey: pagedTasksKey(params),
    queryFn: ({ pageParam, signal }) =>
      listTasks({ ...params, offset: pageParam }, { signal }),
    initialPageParam: 0,
    getNextPageParam: nextOffset,
    // A new search keeps the last answer up until the next one lands.
    placeholderData: keepPreviousData,
    refetchInterval: (query) => pollInterval(tasksOf(query.state.data)),
    // Back to this window: re-read, whatever another tab changed.
    refetchOnWindowFocus: "always",
  });
  useTaskSync();
  const expertsQuery = useListExperts({
    query: { select: (res) => okData(res) ?? [] },
  });
  const tasks = tasksOf(tasksQuery.data);
  const front = openTabs.find((tab) => tab.key === activeKey);

  return {
    tasks,
    experts: expertsQuery.data ?? [],
    filters,
    isLoading: tasksQuery.isLoading,
    isError: tasksQuery.isError && tasksQuery.data === undefined,
    refetch: tasksQuery.refetch,
    hasMore: tasksQuery.hasNextPage,
    isLoadingMore: tasksQuery.isFetchingNextPage,
    loadMore: () => void tasksQuery.fetchNextPage(),
    /** The task open in the sidebar, whose card is marked. */
    selectedID: !isFolded && front?.kind === "task" ? front.taskID : null,
    // A task opens beside the board, in the same sidebar a chat has.
    openTask(id: string) {
      const title = tasks.find((task) => task.id === id)?.title;
      openTab(id, title ?? UNREAD_TITLE);
    },
  };
}

export type HomeBoardState = ReturnType<typeof useHomeBoard>;
