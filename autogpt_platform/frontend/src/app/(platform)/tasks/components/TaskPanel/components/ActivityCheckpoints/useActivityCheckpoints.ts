import { useTaskActivityLog } from "../../../../data/useTasksData";
import { toCheckpoints } from "./helpers";

interface Args {
  taskId: string;
  running: boolean;
  all: boolean;
  resultSummary?: string | null;
}

export function useActivityCheckpoints({
  taskId,
  running,
  all,
  resultSummary,
}: Args) {
  const log = useTaskActivityLog(taskId, { enabled: true });
  const items = log.data?.pages.flatMap((p) => p.items) ?? [];

  return {
    checkpoints: toCheckpoints(items, running, { all, resultSummary }),
    isLoading: log.isLoading,
    hasMore: !!log.hasNextPage,
    isLoadingMore: log.isFetchingNextPage,
    loadMore: () => log.fetchNextPage(),
  };
}
