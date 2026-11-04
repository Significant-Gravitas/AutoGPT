import { useTaskActivityLog } from "../../../../data/useTasksData";
import { toCheckpoints } from "./helpers";

export function useActivityCheckpoints(taskId: string, running: boolean) {
  const log = useTaskActivityLog(taskId, { enabled: true });
  const items = log.data?.pages.flatMap((p) => p.items) ?? [];

  const checkpoints = toCheckpoints(items, running);

  return {
    checkpoints,
    spent: checkpoints.reduce((sum, c) => sum + (c.cost ?? 0), 0),
    isLoading: log.isLoading,
    hasMore: !!log.hasNextPage,
    isLoadingMore: log.isFetchingNextPage,
    loadMore: () => log.fetchNextPage(),
  };
}
