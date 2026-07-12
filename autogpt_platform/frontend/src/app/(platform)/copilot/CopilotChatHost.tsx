"use client";

import { Flask } from "@phosphor-icons/react";
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";
import { ChatContainer } from "./components/ChatContainer/ChatContainer";
import { LocalPCBadge } from "./components/LocalPCBadge/LocalPCBadge";
import { LocalPCComputerUseConsent } from "./components/LocalPCComputerUseConsent/LocalPCComputerUseConsent";
import { RateLimitGate } from "./components/RateLimitResetDialog/RateLimitGate";
import { RecordWorkflow } from "./components/RecordWorkflow/RecordWorkflow";
import { useCopilotPage } from "./useCopilotPage";

interface Props {
  droppedFiles: File[];
  onDroppedFilesConsumed: () => void;
}

/**
 * Session-scoped chat host. Parent mounts this with `key={sessionId}` so
 * session-local view state resets on switch, while the actual AI SDK Chat
 * instance is preserved in the per-session runtime registry.
 */
export function CopilotChatHost({
  droppedFiles,
  onDroppedFilesConsumed,
}: Props) {
  const isRecordingEnabled = useGetFlag(Flag.WORKFLOW_RECORDING);
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
    rateLimitMessage,
    dismissRateLimit,
    sessionDryRun,
    sessionExecutionTarget,
    sessionChatStatus,
  } = useCopilotPage();
  const isLocalSession =
    !!sessionId && sessionExecutionTarget?.kind === "local";

  return (
    <>
      {isLocalSession ? (
        <>
          <div className="flex flex-wrap items-center gap-2 px-4 pt-1.5">
            <LocalPCBadge
              sessionID={sessionId}
              machineID={sessionExecutionTarget.machine_id}
              allowedRoot={sessionExecutionTarget.allowed_root}
            />
            {isRecordingEnabled ? (
              <RecordWorkflow sessionID={sessionId} />
            ) : null}
          </div>
          <LocalPCComputerUseConsent
            key={`local-pc-consent-${sessionId}`}
            sessionID={sessionId}
          />
        </>
      ) : null}
      {/* Only shown when the CURRENT session is confirmed dry_run via its
          immutable metadata. Never based on the global isDryRun preference
          (which only predicts future sessions). */}
      {sessionId && sessionDryRun && (
        <div className="flex items-center justify-center gap-1.5 bg-amber-50 px-3 py-1.5 text-xs font-medium text-amber-800">
          <Flask size={13} weight="bold" />
          Test mode — this session runs agents as simulation
        </div>
      )}
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
      <RateLimitGate
        rateLimitMessage={rateLimitMessage}
        onDismiss={dismissRateLimit}
      />
    </>
  );
}
