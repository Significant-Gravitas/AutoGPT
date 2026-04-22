"use client";

import { Flask } from "@phosphor-icons/react";
import { ChatContainer } from "./components/ChatContainer/ChatContainer";
import { RateLimitGate } from "./components/RateLimitResetDialog/RateLimitGate";
import { useCopilotPage } from "./useCopilotPage";

interface Props {
  droppedFiles: File[];
  onDroppedFilesConsumed: () => void;
}

/**
 * Session-scoped chat host. Parent mounts this with `key={sessionId}` so
 * every session switch gives us a fresh `useChat` Chat instance — no
 * orphaned AI-SDK state bleeding between sessions.
 */
export function CopilotChatHost({
  droppedFiles,
  onDroppedFilesConsumed,
}: Props) {
  const {
    sessionId,
    messages,
    status,
    error,
    stop,
    isReconnecting,
    isSyncing,
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
    historicalDurations,
    rateLimitMessage,
    dismissRateLimit,
    sessionDryRun,
  } = useCopilotPage();

  return (
    <>
      {/* Only shown when the CURRENT session is confirmed dry_run via its
          immutable metadata. Never based on the global isDryRun preference
          (which only predicts future sessions). */}
      {sessionId && sessionDryRun && (
        <div className="flex items-center justify-center gap-1.5 bg-amber-50 px-3 py-1.5 text-xs font-medium text-amber-800">
          <Flask size={13} weight="bold" />
          Test mode — this session runs agents as simulation
        </div>
      )}
      <div className="flex-1 overflow-hidden">
        <ChatContainer
          messages={messages}
          status={status}
          error={error}
          sessionId={sessionId}
          isLoadingSession={isLoadingSession}
          isSessionError={isSessionError}
          isCreatingSession={isCreatingSession}
          isReconnecting={isReconnecting}
          isSyncing={isSyncing}
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
          historicalDurations={historicalDurations}
        />
      </div>
      <RateLimitGate
        rateLimitMessage={rateLimitMessage}
        onDismiss={dismissRateLimit}
      />
    </>
  );
}
