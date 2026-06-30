"use client";

import { ChatContainer } from "@/app/(platform)/copilot/components/ChatContainer/ChatContainer";
import { useTourCopilot } from "./useTourCopilot";

interface Props {
  droppedFiles: File[];
  onDroppedFilesConsumed: () => void;
  onComplete: () => void;
}

export function TourChatHost({
  droppedFiles,
  onDroppedFilesConsumed,
  onComplete,
}: Props) {
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
  } = useTourCopilot({ onComplete });

  return (
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
  );
}
