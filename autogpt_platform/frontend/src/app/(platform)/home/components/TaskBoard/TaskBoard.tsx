"use client";

import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Text } from "@/components/atoms/Text/Text";
import { AddColumn } from "./components/AddColumn/AddColumn";
import { BoardColumn } from "./components/BoardColumn/BoardColumn";
import { ScheduledColumn } from "./components/ScheduledColumn/ScheduledColumn";
import { NewTaskCard } from "./components/NewTaskCard/NewTaskCard";
import { DONE, IN_PROGRESS, isDefaultColumn } from "./helpers";
import { useTaskBoard } from "./useTaskBoard";

export function TaskBoard() {
  const {
    isLoading,
    isClearing,
    clearColumn,
    columns,
    schedules,
    nextFollowupBySession,
    allColumns,
    expertsById,
    moveCard,
    addColumn,
    removeColumn,
  } = useTaskBoard();

  return (
    <section className="relative mt-10 px-4 pb-8 pt-6 text-left md:px-0">
      <div className="relative w-full">
        <header className="px-1 pb-5 pt-1">
          <Text
            variant="lead-semibold"
            as="h2"
            tone="primary"
            className="tracking-[-0.01em]"
          >
            Task board
          </Text>
          <Text variant="body" tone="secondary" className="mt-0.5">
            Every chat is a task. Drag cards between columns to sort them.
          </Text>
        </header>

        {isLoading ? (
          <BoardSkeleton />
        ) : (
          <div className="flex items-start gap-3 overflow-x-auto pb-2">
            <ScheduledColumn schedules={schedules} expertsById={expertsById} />
            {columns.map((column) => (
              <BoardColumn
                key={column.id}
                column={column}
                allColumns={allColumns}
                expertsById={expertsById}
                nextFollowupBySession={nextFollowupBySession}
                onMoveCard={moveCard}
                onRemove={
                  isDefaultColumn(column.id)
                    ? undefined
                    : () => removeColumn(column.id)
                }
                onClear={
                  column.id === DONE && column.sessions.length > 0
                    ? () =>
                        clearColumn(
                          column.sessions.map((session) => session.id),
                        )
                    : undefined
                }
                isClearing={isClearing}
                leading={column.id === IN_PROGRESS ? <NewTaskCard /> : null}
              />
            ))}
            <AddColumn onAdd={addColumn} />
          </div>
        )}
      </div>
    </section>
  );
}

function BoardSkeleton() {
  return (
    <div className="flex gap-3" aria-label="Loading task board…">
      {[0, 1, 2, 3].map((index) => (
        <Skeleton
          key={index}
          className="h-[calc(100dvh-8rem)] min-h-[32rem] min-w-[18rem] max-w-[26rem] flex-1 basis-0 rounded-xl"
        />
      ))}
    </div>
  );
}
