"use client";

import { TaskBoard } from "@/app/(platform)/tasks/components/TaskBoard/TaskBoard";
import { TaskList } from "@/app/(platform)/tasks/components/TaskList/TaskList";
import { LoadingSpinner } from "@/components/atoms/LoadingSpinner/LoadingSpinner";
import { Text } from "@/components/atoms/Text/Text";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { BriefButton } from "../HomeToday/BriefButton";
import { BoardViewSwitch } from "./BoardViewSwitch";
import { CreateTask } from "./CreateTask";
import { useBoardView } from "./useBoardView";
import { useHomeBoard } from "./useHomeBoard";

/** Home's Tasks tab: every task as a board or a list that fills the page,
 *  under a heading row with the layout switch, the day's brief and Create
 *  task. A task opens beside the page, in the sidebar. */
export function HomeBoard() {
  const board = useHomeBoard();
  const { view, setView } = useBoardView();
  const Tasks = view === "list" ? TaskList : TaskBoard;

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
      {board.isError ? (
        <ErrorCard
          context="tasks"
          httpError={{ message: "Your tasks could not be loaded." }}
          onRetry={() => void board.refetch()}
        />
      ) : board.isLoading ? (
        <div className="flex justify-center py-10">
          <LoadingSpinner size="small" />
        </div>
      ) : (
        <Tasks
          tasks={board.tasks}
          experts={board.experts}
          selectedID={board.selectedID}
          onOpen={board.openTask}
        />
      )}
    </section>
  );
}
