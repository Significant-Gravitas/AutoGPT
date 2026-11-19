"use client";

import { Text } from "@/components/atoms/Text/Text";
import { BriefButton } from "../HomeToday/BriefButton";
import { BoardViewSwitch } from "./BoardViewSwitch";
import { BoardBody } from "./components/BoardBody";
import { TaskFiltersBar } from "./components/TaskFiltersBar";
import { CreateTask } from "./CreateTask";
import { useBoardView } from "./useBoardView";
import { useHomeBoard } from "./useHomeBoard";
import { useTaskLink } from "./useTaskLink";

/** Home's Tasks tab: every task as a board or a list that fills the page,
 *  under a heading row with the layout switch, the day's brief and Create
 *  task, and a row to search and filter them. A task opens beside the
 *  page, in the sidebar. */
export function HomeBoard() {
  const board = useHomeBoard();
  const { view, setView } = useBoardView();
  useTaskLink();
  const { filters } = board;

  return (
    <section
      aria-labelledby="home-board-heading"
      className="flex min-h-0 w-full flex-1 flex-col gap-3 text-left"
    >
      <div className="flex items-center gap-2 px-1">
        <Text id="home-board-heading" variant="inbox-medium" as="h2" tone="ink">
          Tasks
        </Text>
        <span className="ml-auto flex items-center gap-2">
          <BoardViewSwitch value={view} onChange={setView} />
          <BriefButton />
          <CreateTask />
        </span>
      </div>
      <TaskFiltersBar
        filters={filters.filters}
        experts={board.experts}
        isFiltered={filters.isFiltered}
        onQuery={filters.setQuery}
        onOwner={filters.setOwner}
        onStatus={filters.setStatus}
        onReset={filters.reset}
      />
      <BoardBody board={board} view={view} />
    </section>
  );
}
