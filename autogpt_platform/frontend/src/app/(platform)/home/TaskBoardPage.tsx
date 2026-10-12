"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { NAVBAR_HEIGHT_PX } from "@/lib/constants";
import { cn } from "@/lib/utils";
import { closestCorners, DndContext, DragOverlay } from "@dnd-kit/core";
import { BoardColumn } from "./components/BoardColumn/BoardColumn";
import { BoardSkeleton } from "./components/BoardSkeleton/BoardSkeleton";
import { AddColumnButton } from "./components/ColumnEditor/AddColumnButton";
import { ColumnMenu } from "./components/ColumnEditor/ColumnMenu";
import { DeleteColumnDialog } from "./components/ColumnEditor/DeleteColumnDialog";
import { useColumnEditor } from "./components/ColumnEditor/useColumnEditor";
import { ReassignDialog } from "./components/ReassignDialog/ReassignDialog";
import { TaskCard } from "./components/TaskCard/TaskCard";
import { TaskPanel } from "./components/TaskPanel/TaskPanel";
import { useTaskBoardPage } from "./useTaskBoardPage";

export function TaskBoardPage() {
  const {
    columns,
    isEmpty,
    isLoading,
    isError,
    errorMessage,
    includeOlder,
    showNewLayout,
    retry,
    toggleOlder,
    openCard,
    drag,
    draggingCard,
    panel,
    closePanel,
    reassignTarget,
    startReassign,
    closeReassign,
    openReassigned,
  } = useTaskBoardPage();
  const editor = useColumnEditor(columns.map(({ column }) => column));

  return (
    <div
      className="flex min-h-0 w-full flex-col"
      style={{
        height: showNewLayout
          ? "100svh"
          : `calc(100vh - ${NAVBAR_HEIGHT_PX}px - var(--preview-banner-height, 0px))`,
      }}
    >
      <div
        className={cn(
          "flex h-16 shrink-0 items-center px-6 md:px-8",
          showNewLayout && "max-md:pl-20",
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
          <DndContext collisionDetection={closestCorners} {...drag}>
            <div className="-mx-6 flex min-h-0 flex-1 items-start gap-3 overflow-x-auto px-6 md:-mx-8 md:px-8">
              {columns.map(({ column, cards }) => (
                <BoardColumn
                  key={column.key}
                  column={column}
                  cards={cards}
                  onOpenCard={openCard}
                  autoExpandAdd={isEmpty && column.key === "backlog"}
                  isRenaming={editor.renamingKey === column.key}
                  onRename={(name) => editor.rename(column.key, name)}
                  onCancelRename={() => editor.cancelRename(column.key)}
                  onDismissRename={editor.dismissRename}
                  menuRef={editor.menuRef(column.key)}
                  menu={
                    <ColumnMenu
                      column={column}
                      canAddColumn={editor.canAddColumn}
                      onRename={() => editor.startRename(column.key)}
                      onAddAfter={() => editor.addColumnAfter(column.key)}
                      onDelete={() => editor.askDelete(column)}
                    />
                  }
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
              {editor.canAddColumn && (
                <AddColumnButton onAdd={editor.addColumn} />
              )}
            </div>
            <DragOverlay>
              {draggingCard && (
                <TaskCard card={draggingCard} onOpen={openCard} isOverlay />
              )}
            </DragOverlay>
          </DndContext>
        )}
      </div>
      {editor.pendingDelete && (
        <DeleteColumnDialog
          column={editor.pendingDelete}
          onConfirm={editor.confirmDelete}
          onCancel={editor.cancelDelete}
        />
      )}
      {panel && (
        <TaskPanel
          key={panel.sessionId}
          card={panel.card}
          sessionId={panel.sessionId}
          columns={panel.columns}
          onClose={closePanel}
          onReassign={startReassign}
        />
      )}
      <ReassignDialog
        key={`reassign-${reassignTarget?.id}`}
        card={reassignTarget}
        onClose={closeReassign}
        onReassigned={openReassigned}
      />
    </div>
  );
}
