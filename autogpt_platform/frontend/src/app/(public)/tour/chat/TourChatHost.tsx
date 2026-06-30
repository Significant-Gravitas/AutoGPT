"use client";

import { ChatContainer } from "@/app/(platform)/copilot/components/ChatContainer/ChatContainer";
import { useState } from "react";
import { TourUpsellModal } from "./components/TourUpsellModal/TourUpsellModal";
import { useTourCopilot } from "./useTourCopilot";

interface Props {
  droppedFiles: File[];
  onDroppedFilesConsumed: () => void;
}

export function TourChatHost({ droppedFiles, onDroppedFilesConsumed }: Props) {
  const [isUpsellOpen, setIsUpsellOpen] = useState(false);
  const {
    sessionId,
    messages,
    status,
    error,
    stop,
    isReconnecting,
    isRestoringActiveSession,
    restoreStatusMessage,
    activeStreamStartedAt,
    isUserStopping,
    createSession,
    onSend,
    onEnqueue,
    queuedMessages,
    isLoadingSession,
    isSessionError,
    isCreatingSession,
    isUploadingFiles,
    hasMoreMessages,
    isLoadingMore,
    loadMore,
    turnStats,
    sessionChatStatus,
    reset,
  } = useTourCopilot({ onComplete: () => setIsUpsellOpen(true) });

  return (
    <>
      <div className="min-h-0 flex-1 overflow-hidden">
        <ChatContainer
          messages={messages}
          status={status}
          error={error}
          sessionId={sessionId}
          sessionChatStatus={sessionChatStatus}
          isLoadingSession={isLoadingSession}
          isSessionError={isSessionError}
          isCreatingSession={isCreatingSession}
          isReconnecting={isReconnecting}
          isRestoringActiveSession={isRestoringActiveSession}
          restoreStatusMessage={restoreStatusMessage}
          activeStreamStartedAt={activeStreamStartedAt}
          isUserStopping={isUserStopping}
          onCreateSession={createSession}
          onSend={onSend}
          onStop={stop}
          onEnqueue={onEnqueue}
          queuedMessages={queuedMessages}
          isUploadingFiles={isUploadingFiles}
          hasMoreMessages={hasMoreMessages}
          isLoadingMore={isLoadingMore}
          onLoadMore={loadMore}
          droppedFiles={droppedFiles}
          onDroppedFilesConsumed={onDroppedFilesConsumed}
          turnStats={turnStats}
        />
      </div>
      <TourUpsellModal
        open={isUpsellOpen}
        onClose={() => setIsUpsellOpen(false)}
        onReplay={() => {
          reset();
          setIsUpsellOpen(false);
        }}
      />
    </>
  );
}
