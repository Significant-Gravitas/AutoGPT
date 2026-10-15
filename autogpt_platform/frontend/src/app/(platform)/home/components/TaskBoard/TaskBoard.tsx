"use client";

import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Text } from "@/components/atoms/Text/Text";
import { AddColumn } from "./components/AddColumn/AddColumn";
import { BoardColumn } from "./components/BoardColumn/BoardColumn";
import { isDefaultColumn } from "./helpers";
import { useTaskBoard } from "./useTaskBoard";

export function TaskBoard() {
  const {
    isLoading,
    columns,
    allColumns,
    expertsById,
    moveCard,
    addColumn,
    removeColumn,
  } = useTaskBoard();

  return (
    <section className="relative mt-10 px-4 pb-8 pt-6 text-left sm:px-6">
      <div className="relative mx-auto w-full max-w-[1120px]">
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
            {columns.map((column) => (
              <BoardColumn
                key={column.id}
                column={column}
                allColumns={allColumns}
                expertsById={expertsById}
                onMoveCard={moveCard}
                onRemove={
                  isDefaultColumn(column.id)
                    ? undefined
                    : () => removeColumn(column.id)
                }
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
      {[0, 1, 2].map((index) => (
        <Skeleton key={index} className="h-64 w-[20rem] shrink-0 rounded-xl" />
      ))}
    </div>
  );
}
