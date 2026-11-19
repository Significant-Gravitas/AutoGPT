"use client";

import { TaskBoard } from "@/app/(platform)/tasks/components/TaskBoard/TaskBoard";
import { TaskList } from "@/app/(platform)/tasks/components/TaskList/TaskList";
import { LoadingSpinner } from "@/components/atoms/LoadingSpinner/LoadingSpinner";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import type { BoardView } from "../useBoardView";
import type { HomeBoardState } from "../useHomeBoard";
import { NoMatches } from "./NoMatches";
import { ShowMore } from "./ShowMore";

interface Props {
  board: HomeBoardState;
  view: BoardView;
}

/** The tasks as a board or a list, what a search that finds nothing says,
 *  and Show more while there are more to read. */
export function BoardBody({ board, view }: Props) {
  if (board.isError) {
    return (
      <ErrorCard
        context="tasks"
        httpError={{ message: "Your tasks could not be loaded." }}
        onRetry={() => void board.refetch()}
      />
    );
  }
  if (board.isLoading) {
    return (
      <div className="flex justify-center py-10">
        <LoadingSpinner size="small" />
      </div>
    );
  }
  if (board.tasks.length === 0 && board.filters.isFiltered) {
    return <NoMatches onReset={board.filters.reset} />;
  }
  const Tasks = view === "list" ? TaskList : TaskBoard;
  return (
    <>
      <Tasks
        tasks={board.tasks}
        experts={board.experts}
        selectedID={board.selectedID}
        onOpen={board.openTask}
      />
      {board.hasMore ? (
        <ShowMore isLoading={board.isLoadingMore} onMore={board.loadMore} />
      ) : null}
    </>
  );
}
