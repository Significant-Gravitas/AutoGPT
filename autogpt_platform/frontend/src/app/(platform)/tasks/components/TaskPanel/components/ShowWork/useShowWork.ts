import { useState } from "react";
import { useTaskActivityLog } from "../../../../data/useTasksData";

export function useShowWork(taskId: string) {
  const [open, setOpen] = useState(false);
  const log = useTaskActivityLog(taskId, { enabled: open });

  return {
    open,
    toggle: () => setOpen((v) => !v),
    items: log.data?.pages.flatMap((p) => p.items) ?? [],
    isLoading: log.isLoading,
    hasMore: !!log.hasNextPage,
    isLoadingMore: log.isFetchingNextPage,
    loadMore: () => log.fetchNextPage(),
  };
}
