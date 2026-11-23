"use client";

import { NewTaskDialog } from "@/app/(platform)/tasks/components/NewTaskDialog/NewTaskDialog";
import { useRef } from "react";
import { BoardBody } from "./components/BoardBody";
import { BoardHeading } from "./components/BoardHeading";
import { SkipToTasks } from "./components/SkipToTasks";
import { TaskFiltersBar } from "./components/TaskFiltersBar";
import { TASKS_REGION_ID, useBoardShortcuts } from "./useBoardShortcuts";
import { useBoardView } from "./useBoardView";
import { useHomeBoard } from "./useHomeBoard";
import { useTaskLauncher } from "./useTaskLauncher";
import { useTaskLink } from "./useTaskLink";

/** Home's Tasks tab: every task as a board or a list that fills the page,
 *  under a heading row (layout, the day's brief, Create task) and a row to
 *  search and filter them. A task opens beside the page, in the sidebar.
 *  N opens Create task and / goes to search. */
export function HomeBoard() {
  const board = useHomeBoard();
  const { view, setView } = useBoardView();
  const launcher = useTaskLauncher();
  const searchRef = useRef<HTMLInputElement>(null);
  const { filters } = board;
  useTaskLink();
  useBoardShortcuts({ onNew: launcher.open, searchRef });

  return (
    <section
      aria-labelledby="home-board-heading"
      className="flex min-h-0 w-full flex-1 flex-col gap-3 text-left"
    >
      <SkipToTasks />
      <BoardHeading view={view} onView={setView} launcher={launcher} />
      <TaskFiltersBar
        searchRef={searchRef}
        filters={filters.filters}
        experts={board.experts}
        isFiltered={filters.isFiltered}
        onQuery={filters.setQuery}
        onOwner={filters.setOwner}
        onStatus={filters.setStatus}
        onSort={filters.setSort}
        onReset={filters.reset}
      />
      <div
        id={TASKS_REGION_ID}
        tabIndex={-1}
        className="flex min-h-0 flex-1 flex-col gap-3 outline-none"
      >
        <BoardBody board={board} view={view} onCreate={launcher.open} />
      </div>
      <NewTaskDialog
        isOpen={launcher.isOpen}
        onOpenChange={launcher.setOpen}
        experts={board.experts}
        initial={launcher.initial}
        onCreated={board.openTask}
        returnFocusRef={launcher.openerRef}
      />
    </section>
  );
}
