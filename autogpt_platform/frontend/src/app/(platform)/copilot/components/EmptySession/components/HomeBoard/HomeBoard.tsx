"use client";

import { TaskBoard } from "@/app/(platform)/tasks/components/TaskBoard/TaskBoard";
import { LoadingSpinner } from "@/components/atoms/LoadingSpinner/LoadingSpinner";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { useHomeBoard } from "./useHomeBoard";

/** The tasks board on Home's Board view; a card opens on the tasks page. */
export function HomeBoard() {
  const board = useHomeBoard();

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
  return (
    <TaskBoard
      tasks={board.tasks}
      experts={board.experts}
      selectedID={null}
      onOpen={board.openTask}
    />
  );
}
