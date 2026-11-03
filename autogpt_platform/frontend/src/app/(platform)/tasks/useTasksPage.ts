import { isComposingEvent } from "@/lib/keyboard";
import { isEditableElement } from "@/lib/platform";
import { parseAsStringLiteral, useQueryState } from "nuqs";
import { useEffect, useState } from "react";
import { useTaskLiveUpdates } from "./data/useTaskLiveUpdates";
import { useTaskExperts, useTaskList } from "./data/useTasksData";
import {
  buildListFilters,
  describeFilters,
  EMPTY_FILTERS,
  isTypingShortcutTarget,
  TASK_VIEWS,
  type PageFilters,
} from "./helpers";

export function useTasksPage() {
  const [view, setView] = useQueryState(
    "view",
    parseAsStringLiteral(TASK_VIEWS).withDefault("board"),
  );
  const [taskID, setTaskID] = useQueryState("task");
  const [filters, setFilters] = useState<PageFilters>(EMPTY_FILTERS);
  const [showHidden, setShowHidden] = useState(false);
  const [isCreateOpenLocal, setIsCreateOpenLocal] = useState(false);
  const [newParam, setNewParam] = useQueryState("new");
  const isCreateOpen = isCreateOpenLocal || newParam === "1";

  function setIsCreateOpen(open: boolean) {
    setIsCreateOpenLocal(open);
    if (!open && newParam) setNewParam(null);
  }

  useTaskLiveUpdates({ withToasts: false });

  const tasksQuery = useTaskList(
    buildListFilters({ filters, showHidden, view }),
  );
  const expertsQuery = useTaskExperts();
  const experts = expertsQuery.data ?? [];

  useEffect(() => {
    function handleKeyDown(e: KeyboardEvent) {
      if (isComposingEvent(e) || e.metaKey || e.ctrlKey || e.altKey) return;
      if (e.key.toLowerCase() !== "n") return;
      const active = document.activeElement;
      if (isEditableElement(active) || isTypingShortcutTarget(active)) return;
      e.preventDefault();
      setIsCreateOpenLocal(true);
    }
    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, []);

  function openTask(id: string) {
    setTaskID(id);
  }

  function handleSheetOpenChange(open: boolean) {
    if (!open) setTaskID(null);
  }

  return {
    view,
    setView,
    taskID,
    openTask,
    handleSheetOpenChange,
    filters,
    setFilters,
    filterChips: describeFilters(filters, experts),
    showHidden,
    setShowHidden,
    isCreateOpen,
    setIsCreateOpen,
    experts,
    tasks: tasksQuery.data ?? [],
    isLoading: tasksQuery.isLoading,
    error: tasksQuery.error,
    refetch: tasksQuery.refetch,
  };
}
