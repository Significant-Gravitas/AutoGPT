"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { NAVBAR_HEIGHT_PX } from "@/lib/constants";
import { cn } from "@/lib/utils";
import { BoardColumn } from "./components/BoardColumn/BoardColumn";
import { BoardSkeleton } from "./components/BoardSkeleton/BoardSkeleton";
import { useTaskBoardPage } from "./useTaskBoardPage";

export function TaskBoardPage() {
  const {
    columns,
    isEmpty,
    isLoading,
    isError,
    errorMessage,
    includeOlder,
    isNewLayout,
    retry,
    toggleOlder,
    openCard,
  } = useTaskBoardPage();

  return (
    <div
      className="flex min-h-0 w-full flex-col"
      style={{
        height: isNewLayout
          ? "100svh"
          : `calc(100vh - ${NAVBAR_HEIGHT_PX}px - var(--preview-banner-height, 0px))`,
      }}
    >
      <div
        className={cn(
          "flex h-16 shrink-0 items-center px-6 md:px-8",
          isNewLayout && "max-md:pl-20",
        )}
      >
        <Text variant="h4" as="h1">
          Board
        </Text>
      </div>
      <div className="flex min-h-0 flex-1 flex-col px-6 pb-6 md:px-8">
        {isLoading && <BoardSkeleton />}
        {isError && (
          <ErrorCard
            context="your board"
            httpError={{ message: errorMessage }}
            onRetry={retry}
            className="max-w-xl"
          />
        )}
        {columns.length > 0 && (
          <div className="-mx-6 flex min-h-0 flex-1 items-start gap-3 overflow-x-auto px-6 md:-mx-8 md:px-8">
            {columns.map(({ column, cards }) => (
              <BoardColumn
                key={column.key}
                column={column}
                cards={cards}
                onOpenCard={openCard}
                autoExpandAdd={isEmpty && column.key === "backlog"}
                footer={
                  column.key === "done" && (
                    <Button
                      variant="ghost"
                      size="xs"
                      onClick={toggleOlder}
                      className="self-center text-zinc-500"
                    >
                      {includeOlder ? "Hide older" : "Show older"}
                    </Button>
                  )
                }
              />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
