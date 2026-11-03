import { isComposingEvent, isKey } from "@/lib/keyboard";
import { isEditableElement } from "@/lib/platform";
import { parseAsBoolean, parseAsStringLiteral, useQueryState } from "nuqs";
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
} from "./helpers";

export function useTasksPage() {
  const [view, setView] = useQueryState(
    "view",
    parseAsStringLiteral(TASK_VIEWS).withDefault("list"),
  );
  const [tab, setTab] = useQueryState(
    "tab",
    parseAsStringLiteral(INBOX_TABS).withDefault("all"),
  );
  const [mine, setMine] = useQueryState(
    "mine",
    parseAsBoolean.withDefault(false),
  );
  const [taskID, setTaskID] = useQueryState("task");
  const [filters, setFilters] = useState<PageFilters>(EMPTY_FILTERS);
  const [expandedID, setExpandedID] = useState<string | null>(null);
  const create = useCreateDialog();

  useTaskLiveUpdates({ withToasts: false });

  const inboxQuery = useTaskInbox(buildListFilters({ filters, mine }));
  const expertsQuery = useTaskExperts();
  const experts = expertsQuery.data ?? [];
  const allTasks = inboxQuery.data?.tasks ?? [];

  useCollapseOnEscape({
    expandedID,
    isSheetOpen: !!taskID,
    onCollapse: setExpandedID,
  });

  function toggleRow(id: string) {
    setExpandedID((current) => (current === id ? null : id));
  }

  function clearFilters() {
    setFilters(EMPTY_FILTERS);
    setMine(false);
  }

  function openTask(id: string) {
    setTaskID(id);
  }

  function handleSheetOpenChange(open: boolean) {
    if (!open) setTaskID(null);
  }

  return {
    view,
    setView,
    tab,
    setTab,
    mine,
    setMine,
    taskID,
    openTask,
    handleSheetOpenChange,
    expandedID,
    toggleRow,
    filters,
    setFilters,
    clearFilters,
    filterChips: describeFilters(filters, experts),
    experts,
    tasks: filterInbox(allTasks, tab),
    counts: inboxQuery.data?.counts,
    isLoading: inboxQuery.isLoading,
    error: inboxQuery.error,
    refetch: inboxQuery.refetch,
    ...create,
  };
}

function useCreateDialog() {
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
    if (!open && newParam) setNewParam(null);
  }

  return {
    isCreateOpen: isCreateOpenLocal || newParam === "1",
    setIsCreateOpen,
  };
}

function useCollapseOnEscape(args: {
  expandedID: string | null;
  isSheetOpen: boolean;
  onCollapse: (id: null) => void;
}) {
  const { expandedID, isSheetOpen, onCollapse } = args;

  useEffect(() => {
    if (!expandedID || isSheetOpen) return;
    function handleKeyDown(e: KeyboardEvent) {
      if (!isKey(e, "Escape") || e.defaultPrevented) return;
      if (isTypingShortcutTarget(document.activeElement)) return;
      onCollapse(null);
      document.getElementById(inboxRowID(expandedID!))?.focus();
    }
    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [expandedID, isSheetOpen, onCollapse]);
}
