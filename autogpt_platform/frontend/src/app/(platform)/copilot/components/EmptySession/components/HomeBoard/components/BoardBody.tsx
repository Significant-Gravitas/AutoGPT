"use client";

import { TaskBoard } from "@/app/(platform)/tasks/components/TaskBoard/TaskBoard";
import { TaskList } from "@/app/(platform)/tasks/components/TaskList/TaskList";
import type { TaskDraft } from "@/app/(platform)/tasks/components/NewTaskDialog/draft";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import type { BoardView } from "../useBoardView";
import type { HomeBoardState } from "../useHomeBoard";
import { BoardSkeleton } from "./BoardSkeleton";
import { FirstTasks } from "./FirstTasks";
import { NoMatches } from "./NoMatches";
import { ShowMore } from "./ShowMore";

interface Props {
  board: HomeBoardState;
  view: BoardView;
  onCreate: (from?: TaskDraft) => void;
}

/** The tasks as a board or a list (a skeleton of either on the first
 *  read), the empty board's first steps, what a search that finds nothing
 *  says, and Show more while there are more to read. */
export function BoardBody({ board, view, onCreate }: Props) {
  if (board.isError) {
    return (
      <ErrorCard
        context="tasks"
        httpError={{ message: "Your tasks could not be loaded." }}
        onRetry={() => void board.refetch()}
      />
    );
  }
  if (board.isLoading) return <BoardSkeleton view={view} />;
  if (board.tasks.length === 0) {
    return board.filters.isFiltered ? (
      <NoMatches onReset={board.filters.reset} />
    ) : (
      <FirstTasks experts={board.experts} onCreate={onCreate} />
    );
  }
  const Tasks = view === "list" ? TaskList : TaskBoard;
  return (
    <>
      <Tasks
        tasks={board.tasks}
        experts={board.experts}
        selectedID={board.selectedID}
        onOpen={board.openTask}
        keepOrder={board.filters.isSorted}
      />
      {board.hasMore ? (
        <ShowMore isLoading={board.isLoadingMore} onMore={board.loadMore} />
      ) : null}
    </>
  );
}
