"use client";
import { ChatContainer } from "./components/ChatContainer/ChatContainer";
import type { FloatingControls } from "./components/ChatMessagesContainer/components/ThreadHeader";
import { ProviderLimitDialog } from "./components/ProviderLimitDialog/ProviderLimitDialog";
import { RateLimitGate } from "./components/RateLimitResetDialog/RateLimitGate";
import { useCopilotPage, useCopilotRuntimePage } from "./useCopilotPage";
import { FlaskConicalIcon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";
import { useUsageExperience } from "@/services/usageExperience/useUsageExperience";
import { useRateLimitRefresh } from "./components/RateLimitResetDialog/useRateLimitRefresh";

interface Props {
  droppedFiles: File[];
  onDroppedFilesConsumed: () => void;
  /** Which layout floats its controls over the chat's top-left corner on
   *  small viewports. */
  floatingControls?: FloatingControls;
}

type CopilotPage = ReturnType<typeof useCopilotPage>;

/**
 * Session-scoped chat host. Parent mounts this with `key={sessionId}` so
 * session-local view state resets on switch, while the stream itself is
 * preserved in a per-session runtime registry. Keyed again by the stream
 * flag, so flags resolving after mount swap the stream hook by remounting.
 */
export function CopilotChatHost(props: Props) {
  const isStreamRuntime = useGetFlag(Flag.COPILOT_STREAM_RUNTIME) === true;
  return isStreamRuntime ? (
    <RuntimeChatHostBody key="stream-runtime" {...props} />
  ) : (
    <AiSdkChatHostBody key="ai-sdk" {...props} />
  );
}

function AiSdkChatHostBody(props: Props) {
  const page = useCopilotPage();
  const usage = useUsageExperience();
  const usageRefresh = useRateLimitRefresh(page.rateLimitMessage, usage.retry);
  return (
    <ChatHostView
      {...props}
      page={page}
      usage={usage}
      usageRefresh={usageRefresh}
    />
  );
}

function RuntimeChatHostBody(props: Props) {
  const page = useCopilotRuntimePage();
  const usage = useUsageExperience();
  const usageRefresh = useRateLimitRefresh(page.rateLimitMessage, usage.retry);
  return (
    <ChatHostView
      {...props}
      page={page}
      usage={usage}
      usageRefresh={usageRefresh}
    />
  );
}

function ChatHostView({
  droppedFiles,
  onDroppedFilesConsumed,
  floatingControls,
  page,
  usage,
  usageRefresh,
}: Props & {
  page: CopilotPage;
  usage: ReturnType<typeof useUsageExperience>;
  usageRefresh: ReturnType<typeof useRateLimitRefresh>;
}) {
  const {
    sessionId,
    messages,
    status,
    error,
    stop,
    isReconnecting,
    isFinishProbing,
    isRestoringActiveSession,
    restoreStatusMessage,
    activeStreamStartedAt,
    isUserStopping,
    createSession,
    onSend,
    onStartTask,
    onEnqueue,
    queuedMessages,
    isLoadingSession,
    isSessionError,
    isSessionNotFound,
    isCreatingSession,
    isUploadingFiles,
    pendingSend,
    hasMoreMessages,
    isLoadingMore,
    loadMore,
    turnStats,
    rateLimitMessage,
    platformLimitFailure,
    dismissRateLimit,
    providerLimit,
    dismissProviderLimit,
    sessionDryRun,
    sessionChatStatus,
    sessionSentFrom,
    sessionAutopilotMode,
    chatLlmAuthProvider,
    expertIdentity,
    isResolvingExpertIdentity,
    isAdoptingExpertSession,
    isKickoffStarting,
    followBackendTurn,
  } = page;
  function dismissAfterProviderChange() {
    usageRefresh.release();
    dismissRateLimit();
  }

  return (
    <>
      {/* Only shown when the CURRENT session is confirmed dry_run via its
          immutable metadata. Never based on the global isDryRun preference
          (which only predicts future sessions). */}
      {/* TODO(tokens): no exact role for amber-800 */}
      {sessionId && sessionDryRun && (
        <div className="flex items-center justify-center gap-1.5 bg-warning-surface px-3 py-1.5 text-xs font-medium text-amber-800">
          <Icon icon={FlaskConicalIcon} size={13} />
          Test mode — this session runs workflows as simulation
        </div>
      )}
      <div className="min-h-0 flex-1 overflow-hidden">
        <ChatContainer
          isUsageRefused={!!rateLimitMessage || usageRefresh.guarded}
          usageFailure={platformLimitFailure}
          onDismissUsage={dismissAfterProviderChange}
          usageRefresh={usageRefresh}
          usage={usage}
          chatAuthProvider={chatLlmAuthProvider}
          messages={messages}
          status={status}
          error={error}
          sessionId={sessionId}
          sessionChatStatus={sessionChatStatus}
          sessionSentFrom={sessionSentFrom}
          sessionAutopilotMode={sessionAutopilotMode}
          isLoadingSession={isLoadingSession}
          isSessionError={isSessionError}
          isSessionNotFound={isSessionNotFound}
          isCreatingSession={isCreatingSession}
          isReconnecting={isReconnecting}
          isFinishProbing={isFinishProbing}
          isRestoringActiveSession={isRestoringActiveSession}
          restoreStatusMessage={restoreStatusMessage}
          activeStreamStartedAt={activeStreamStartedAt}
          isUserStopping={isUserStopping}
          onCreateSession={createSession}
          onSend={onSend}
          onStartTask={onStartTask}
          onStop={stop}
          onEnqueue={onEnqueue}
          queuedMessages={queuedMessages}
          isUploadingFiles={isUploadingFiles}
          pendingSend={pendingSend}
          hasMoreMessages={hasMoreMessages}
          isLoadingMore={isLoadingMore}
          onLoadMore={loadMore}
          droppedFiles={droppedFiles}
          onDroppedFilesConsumed={onDroppedFilesConsumed}
          turnStats={turnStats}
          expertIdentity={expertIdentity}
          isResolvingExpertIdentity={isResolvingExpertIdentity}
          isAdoptingExpertSession={isAdoptingExpertSession}
          isKickoffStarting={isKickoffStarting}
          onBackendTurn={followBackendTurn}
          floatingControls={floatingControls}
        />
      </div>
      <RateLimitGate
        refreshState={usageRefresh}
        rateLimitMessage={rateLimitMessage}
        failure={platformLimitFailure}
        sessionId={sessionId}
        onDismiss={dismissRateLimit}
        onSendNow={onSend}
      />
      <ProviderLimitDialog
        failure={providerLimit}
        sessionId={sessionId}
        onDismiss={dismissProviderLimit}
      />
    </>
  );
}
