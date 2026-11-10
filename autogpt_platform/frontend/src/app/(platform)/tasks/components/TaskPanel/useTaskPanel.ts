import { useGetTask } from "@/app/api/__generated__/endpoints/tasks/tasks";
import type { Expert } from "@/app/api/__generated__/models/expert";
import { useEffect, useState } from "react";
import { isSettled, ownerOf, POLL_MS } from "../../helpers";

export const PANEL_TITLE_ID = "task-panel-title";

export function useTaskPanel(taskID: string, experts: Expert[]) {
  const query = useGetTask(taskID, {
    query: {
      select: (res) => (res.status === 200 ? res.data : null),
      refetchInterval: (q) => {
        const res = q.state.data;
        if (res?.status !== 200) return false;
        return isSettled(res.data.status) ? false : POLL_MS;
      },
    },
  });
  const task = query.data ?? null;
  const [isRenaming, setIsRenaming] = useState(false);
  const hasTask = !!task;

  // Opening a task moves focus to its title, so the panel is read first.
  useEffect(() => {
    if (hasTask) document.getElementById(PANEL_TITLE_ID)?.focus();
  }, [hasTask]);

  return {
    task,
    owner: task ? ownerOf(task, experts) : null,
    isLoading: query.isLoading,
    isError: query.isError || (!query.isLoading && !task),
    retry: () => void query.refetch(),
    isRenaming,
    startRename: () => setIsRenaming(true),
    stopRename: () => setIsRenaming(false),
  };
}
