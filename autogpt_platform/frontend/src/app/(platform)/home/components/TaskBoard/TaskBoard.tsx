"use client";

import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Text } from "@/components/atoms/Text/Text";
import { AddColumn } from "./components/AddColumn/AddColumn";
import { BoardColumn } from "./components/BoardColumn/BoardColumn";
import { ScheduledColumn } from "./components/ScheduledColumn/ScheduledColumn";
import { NEW_SCHEDULED_TASK_PROMPT } from "@/components/contextual/guidedPrompts";
import { ComposerCtaCard } from "./components/ComposerCtaCard/ComposerCtaCard";
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
          <div className="mt-0.5 flex items-center justify-between gap-4">
            <Text variant="body" tone="secondary">
              Every chat is a task. Drag cards between columns to sort them.
            </Text>
            {!isLoading && <AddColumn onAdd={addColumn} />}
          </div>
        </header>

        {isLoading ? (
          <BoardSkeleton />
        ) : (
          <div className="flex items-start gap-3 overflow-x-auto pb-2">
            <ScheduledColumn
              schedules={schedules}
              expertsById={expertsById}
              leading={
                <ComposerCtaCard
                  title="New schedule"
                  description="Ask Otto to run something on a schedule"
                  prompt={NEW_SCHEDULED_TASK_PROMPT}
                />
              }
            />
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
                leading={
                  column.id === IN_PROGRESS ? (
                    <ComposerCtaCard
                      title="New task"
                      description="Tell Otto or an expert what to do"
                    />
                  ) : null
                }
              />
            ))}
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
