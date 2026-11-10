import { useListExperts } from "@/app/api/__generated__/endpoints/experts/experts";
import { useListTasks } from "@/app/api/__generated__/endpoints/tasks/tasks";
import { isKey } from "@/lib/keyboard";
import { isEditableElement } from "@/lib/platform";
import { parseAsStringLiteral, useQueryState } from "nuqs";
import { useEffect, useState } from "react";
import {
  filterTasks,
  pollInterval,
  TASK_TABS,
  TASK_VIEWS,
  TASKS_HEADING_ID,
  taskRowID,
} from "./helpers";

/** The API's page size cap: the page shows the newest hundred tasks. */
const LIST_LIMIT = 100;

export function useTasksPage() {
  const [view, setView] = useQueryState(
    "view",
    parseAsStringLiteral(TASK_VIEWS).withDefault("list"),
  );
  const [tab, setTab] = useQueryState(
    "tab",
    parseAsStringLiteral(TASK_TABS).withDefault("all"),
  );
  const [taskID, setTaskID] = useQueryState("task");
  const [isCreateOpen, setIsCreateOpen] = useState(false);

  const tasksQuery = useListTasks(
    { limit: LIST_LIMIT },
    {
      query: {
        select: (res) => (res.status === 200 ? res.data.tasks : []),
        refetchInterval: (query) => {
          const res = query.state.data;
          return pollInterval(res?.status === 200 ? res.data.tasks : undefined);
        },
      },
    },
  );
  const expertsQuery = useListExperts({
    query: { select: (res) => (res.status === 200 ? res.data : []) },
  });

  const allTasks = tasksQuery.data ?? [];

  function openTask(id: string) {
    void setTaskID(id);
  }

  function closeTask() {
    if (!taskID) return;
    const rowID = taskRowID(taskID);
    void setTaskID(null);
    // The row is gone after an archive; the page heading takes focus then.
    requestAnimationFrame(() =>
      (
        document.getElementById(rowID) ??
        document.getElementById(TASKS_HEADING_ID)
      )?.focus(),
    );
  }

  useEscapeToClose({ isOpen: !!taskID, onClose: closeTask });

  return {
    view,
    setView,
    tab,
    setTab,
    taskID,
    openTask,
    closeTask,
    allTasks,
    tasks: filterTasks(allTasks, tab),
    experts: expertsQuery.data ?? [],
    isLoading: tasksQuery.isLoading,
    error: tasksQuery.error,
    hasData: tasksQuery.data !== undefined,
    refetch: tasksQuery.refetch,
    isCreateOpen,
    setIsCreateOpen,
  };
}

function useEscapeToClose(args: { isOpen: boolean; onClose: () => void }) {
  const { isOpen, onClose } = args;

  useEffect(() => {
    if (!isOpen) return;
    function handleKeyDown(e: KeyboardEvent) {
      if (!isKey(e, "Escape") || e.defaultPrevented) return;
      const active = document.activeElement;
      if (isEditableElement(active)) return;
      if (active?.closest("[role='dialog'], [role='menu']")) return;
      onClose();
    }
    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [isOpen, onClose]);
}
