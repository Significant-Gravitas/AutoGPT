import { isComposingEvent, isKey } from "@/lib/keyboard";
import { isEditableElement } from "@/lib/platform";
import { parseAsStringLiteral, useQueryState } from "nuqs";
import { useEffect, useState } from "react";
import { useTaskLiveUpdates } from "./data/useTaskLiveUpdates";
import { useTaskExperts, useTaskInbox } from "./data/useTasksData";
import {
  buildListFilters,
  describeFilters,
  EMPTY_FILTERS,
  filterInbox,
  INBOX_TABS,
  inboxRowID,
  isTypingShortcutTarget,
  TASK_VIEWS,
  type PageFilters,
  type TasksView,
} from "./helpers";

interface Options {
  defaultView?: TasksView;
  openCreateFromURL?: boolean;
}

export function useTasksPage({
  defaultView = "list",
  openCreateFromURL = true,
}: Options = {}) {
  const [view, setView] = useQueryState(
    "view",
    parseAsStringLiteral(TASK_VIEWS).withDefault(defaultView),
  );
  const [tab, setTab] = useQueryState(
    "tab",
    parseAsStringLiteral(INBOX_TABS).withDefault("all"),
  );
  const [taskID, setTaskID] = useQueryState("task");
  const [filters, setFilters] = useState<PageFilters>(EMPTY_FILTERS);
  const create = useCreateDialog(openCreateFromURL);

  useTaskLiveUpdates({ withToasts: false });

  const inboxQuery = useTaskInbox(buildListFilters(filters));
  const expertsQuery = useTaskExperts();
  const experts = expertsQuery.data ?? [];

  function closeTask() {
    if (!taskID) return;
    const rowID = inboxRowID(taskID);
    setTaskID(null);
    requestAnimationFrame(() => document.getElementById(rowID)?.focus());
  }

  useInboxEscape({ selectedID: taskID, onDeselect: closeTask });

  function clearFilters() {
    setFilters(EMPTY_FILTERS);
  }

  function openTask(id: string) {
    setTaskID(id);
  }

  return {
    view,
    setView,
    tab,
    setTab,
    taskID,
    openTask,
    closeTask,
    filters,
    setFilters,
    clearFilters,
    filterChips: describeFilters(filters, experts),
    experts,
    tasks: filterInbox(inboxQuery.data?.tasks ?? [], tab),
    counts: inboxQuery.data?.counts,
    isLoading: inboxQuery.isLoading,
    error: inboxQuery.error,
    refetch: inboxQuery.refetch,
    ...create,
  };
}

function useCreateDialog(openFromURL: boolean) {
  const [isCreateOpenLocal, setIsCreateOpenLocal] = useState(false);
  const [newParam, setNewParam] = useQueryState("new");

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

  function setIsCreateOpen(open: boolean) {
    setIsCreateOpenLocal(open);
    if (!open && openFromURL && newParam) setNewParam(null);
  }

  return {
    isCreateOpen: isCreateOpenLocal || (openFromURL && newParam === "1"),
    setIsCreateOpen,
  };
}

function useInboxEscape(args: {
  selectedID: string | null;
  onDeselect: () => void;
}) {
  const { selectedID, onDeselect } = args;

  useEffect(() => {
    if (!selectedID) return;
    function handleKeyDown(e: KeyboardEvent) {
      if (!isKey(e, "Escape") || e.defaultPrevented) return;
      const active = document.activeElement;
      if (isEditableElement(active) || isTypingShortcutTarget(active)) return;
      onDeselect();
    }
    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [selectedID, onDeselect]);
}
