"use client";

import { CopilotChatHost } from "@/app/(platform)/copilot/CopilotChatHost";
import { ContextPanelAutoOpen } from "@/app/(platform)/copilot/components/ContextPanel/ContextPanelAutoOpen";
import { CopilotModals } from "@/app/(platform)/copilot/components/CopilotModals/CopilotModals";
import { WorkspaceFileCards } from "@/app/(platform)/copilot/components/WorkspaceFileCards/WorkspaceFileCards";
import type { BoardColumn } from "@/app/api/__generated__/models/boardColumn";
import { Sheet, SheetContent } from "@/components/ui/sheet";
import dynamic from "next/dynamic";
import type { Card } from "../../board-helpers";
import { DraftStart } from "./components/DraftStart/DraftStart";
import { StartBar } from "./components/StartBar/StartBar";
import { TaskPanelHeader } from "./components/TaskPanelHeader/TaskPanelHeader";
import { useTaskPanel } from "./useTaskPanel";

const ArtifactPanel = dynamic(
  () =>
    import(
      "@/app/(platform)/copilot/components/ArtifactPanel/ArtifactPanel"
    ).then((m) => m.ArtifactPanel),
  { ssr: false },
);

interface Props {
  card: Card | null;
  sessionId: string;
  columns: BoardColumn[];
  onClose: () => void;
  onReassign: (card: Card) => void;
}

function noop() {}

export function TaskPanel({
  card,
  sessionId,
  columns,
  onClose,
  onReassign,
}: Props) {
  const {
    title,
    columnKey,
    isDraft,
    canStart,
    isStarting,
    showFiles,
    toggleFiles,
    handleRename,
    handleAssign,
    handleMove,
    handleStart,
    handleStartWithTitle,
    handleEscapeKeyDown,
  } = useTaskPanel({ card, sessionId, columns });

  function handleOpenChange(open: boolean) {
    if (!open) onClose();
  }

  return (
    <Sheet open onOpenChange={handleOpenChange}>
      <SheetContent
        side="right"
        aria-describedby={undefined}
        onEscapeKeyDown={handleEscapeKeyDown}
        className="flex w-1/2 min-w-[560px] max-w-none flex-col gap-0 p-0 max-md:w-full max-md:min-w-0 sm:max-w-none"
      >
        <TaskPanelHeader
          sessionId={sessionId}
          title={title}
          card={card}
          columns={columns}
          columnKey={columnKey}
          showFiles={showFiles}
          onToggleFiles={toggleFiles}
          onRename={handleRename}
          onAssign={handleAssign}
          onMove={handleMove}
          onReassign={onReassign}
        />
        <div className="flex min-h-0 flex-1 flex-col bg-[#f6f7fb]">
          {showFiles && (
            <div className="min-h-0 flex-1 overflow-y-auto p-4">
              <WorkspaceFileCards sessionId={sessionId} />
            </div>
          )}
          {!showFiles && isDraft && card && (
            <DraftStart
              key={card.id}
              cardId={card.id}
              initialMessage={card.title ?? ""}
              isStarting={isStarting}
              onStart={handleStart}
            />
          )}
          {!showFiles && canStart && (
            <StartBar isStarting={isStarting} onStart={handleStartWithTitle} />
          )}
          {!showFiles && !isDraft && (
            <CopilotChatHost
              key={sessionId}
              droppedFiles={[]}
              onDroppedFilesConsumed={noop}
            />
          )}
        </div>
        <ContextPanelAutoOpen sessionId={sessionId} />
        <ArtifactPanel mobile />
        <CopilotModals />
      </SheetContent>
    </Sheet>
  );
}
