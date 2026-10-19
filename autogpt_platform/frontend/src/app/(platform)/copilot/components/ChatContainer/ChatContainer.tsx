"use client";
import { ChatInput } from "@/app/(platform)/copilot/components/ChatInput/ChatInput";
import { ChatComposerFrame } from "./components/ChatComposerFrame";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/atoms/Tooltip/BaseTooltip";
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";
import { UIDataTypes, UIMessage, UITools } from "ai";
import { LayoutGroup } from "framer-motion";
import { useCallback, useEffect, useRef, useState } from "react";
import type { TurnStatsMap } from "../../helpers/convertChatSessionToUiMessages";
import type { WorkspaceAttachment } from "../../helpers/workspaceAttachments";
import type { PendingUploadSend } from "../../copilotStreamStore";
import { ChatMessagesContainer } from "../ChatMessagesContainer/ChatMessagesContainer";
import type { FloatingControls } from "../ChatMessagesContainer/components/ThreadHeader";
import { CopilotChatActionsProvider } from "../CopilotChatActionsProvider/CopilotChatActionsProvider";
import { NewChatOnboarding } from "../ExpertOnboardingCard/NewChatOnboarding";
import { EmptySession } from "../EmptySession/EmptySession";
import { UsageLimitReachedCard } from "../UsageLimits/UsageLimitReachedCard/UsageLimitReachedCard";
import type { UsageNoticeContext } from "../UsageLimits/UsageLimitReachedCard/useUsageLimitReachedCard";
import { useIsUsageLimitReached } from "../UsageLimits/useIsUsageLimitReached";
import { BudgetLowHint } from "../UsageLimits/BudgetLowHint";
import { ScheduledSendChip } from "../SendAtReset/ScheduledSendChip";
import { SESSION_COMPOSER_ID, sessionDraftKey } from "../SendAtReset/helpers";
import { useComposerBudget } from "./useComposerBudget";
import type { UsageSnapshot } from "@/services/usageExperience/useUsageExperience";
import { TaskProgressBar } from "../TaskProgressBar/TaskProgressBar";
import { getLatestTaskList } from "../TaskProgressBar/helpers";
import { ContextPanelToggle } from "../ContextPanel/ContextPanelToggle";
import { useShareChatExpert } from "../ContextPanel/useShareChatExpert";
import { ArchivedExpertNotice } from "./components/ArchivedExpertNotice";
import { SessionNotFound } from "./components/SessionNotFound";
import { cn } from "@/lib/utils";
import { useIsMobile } from "../../useIsMobile";
import { ComposerPanelTab } from "./components/ComposerPanelTab";
import { SharedChatNotice } from "./components/SharedChatNotice";
import { useAutoOpenArtifacts } from "./useAutoOpenArtifacts";
import { VoiceModeBar } from "../../voice/components/VoiceModeBar";
import { VoiceModeButton } from "../../voice/components/VoiceModeButton";
import { useVoiceMode } from "../../voice/useVoiceMode";
import { useVoiceSilenceTimeout } from "../../voice/useVoiceSilenceTimeout";
import {
  requestVoiceStart,
  takeVoiceStart,
} from "../../voice/pendingVoiceStart";
import { unlockAudio } from "../../voice/speechPlayer";
import type { ExpertIdentity } from "../../useExpertMap";
import { isTokenDevtoolEnabled } from "../../tokenDevtool/gate";
import { updateHistoryBreakdown } from "../../tokenDevtool/store";
import { breakdownCacheKey } from "../../tokenDevtool/tokenMath";
import type { SentFrom } from "../../sentFrom";
import type { AutopilotMode } from "../../autopilotModeStore";
import { AutopilotModeSelector } from "../ChatInput/components/AutopilotModeSelector/AutopilotModeSelector";
import { toastSendFailure } from "../../helpers/sendFailureToast";
import { isHeldCallRow } from "../ChatMessagesContainer/heldCallRows";
import {
  getKickoffAttemptToken,
  getKickoffExpertId,
  stripLegacyKickoffMarker,
  type ExpertKickoffMetadata,
} from "../../expertKickoff";

export interface ChatContainerProps {
  isUsageRefused?: boolean;
  usageFailure?: import("../../providerFailure").ProviderFailure | null;
  onDismissUsage?: () => void;
  usageRefresh?: UsageNoticeContext["refresh"];
  /** The account's AutoPilot budget: drives Send at reset and the
   *  running-low hint. */
  usage?: UsageSnapshot | null;
  /** The connection the chat runs on, or for a new chat the one it will
   *  start on. Only a platform chat waits for the AutoPilot budget. */
  chatAuthProvider?: string | null;
  messages: UIMessage<unknown, UIDataTypes, UITools>[];
  status: string;
  error: Error | undefined;
  sessionId: string | null;
  sessionChatStatus?: string;
  sessionSentFrom?: SentFrom | null;
  /** The approval mode stored on the session, when it has one. */
  sessionAutopilotMode?: AutopilotMode | null;
  isLoadingSession: boolean;
  isSessionError?: boolean;
  isSessionNotFound?: boolean;
  isCreatingSession: boolean;
  /** True when backend has an active stream but we haven't reconnected yet. */
  isReconnecting?: boolean;
  /** True while a closed stream is being checked for a turn still running. */
  isFinishProbing?: boolean;
  /** True while reopening an already-running session before stream replay is live. */
  isRestoringActiveSession?: boolean;
  /** Latest backend-emitted status for a replaying assistant while restore is active. */
  restoreStatusMessage?: string | null;
  /** ISO start time of the active backend turn (if any). Seeds the elapsed
   * counter for restored sessions so the UI shows honest turn age. */
  activeStreamStartedAt?: string | null;
  /** True the moment the user clicks Stop — overrides isStreaming so the UI
   * flips immediately regardless of AI SDK's status timing. */
  isUserStopping?: boolean;
  onCreateSession: () => void | Promise<string>;
  onSend: (
    message: string,
    files?: File[],
    workspaceFiles?: WorkspaceAttachment[],
    metadata?: ExpertKickoffMetadata,
  ) => void | Promise<void>;
  onStop: () => void;
  /** Called to enqueue a message while streaming (bypasses normal send flow). */
  onEnqueue?: (message: string) => void | Promise<void>;
  /** Pending queued messages waiting to be injected, shown at the end of chat. */
  queuedMessages?: string[];
  isUploadingFiles?: boolean;
  /** The message whose attachments are still uploading, shown as a
   *  placeholder bubble until the real one lands in `messages`. */
  pendingSend?: PendingUploadSend | null;
  hasMoreMessages?: boolean;
  isLoadingMore?: boolean;
  onLoadMore?: () => void;
  /** Files dropped onto the chat window. */
  droppedFiles?: File[];
  /** Called after droppedFiles have been consumed by ChatInput. */
  onDroppedFilesConsumed?: () => void;
  /** Per-message stats (durationMs, createdAt), keyed by message ID. */
  turnStats?: TurnStatsMap;
  /** Expert identity for expert-scoped sessions (thread header + assistant
   * avatar/name). Null = default header. */
  expertIdentity?: ExpertIdentity | null;
  /** True while an expert-scoped session's active/archived identity is still
   * unresolved. Keep every send path locked until the roster settles. */
  isResolvingExpertIdentity?: boolean;
  /** True while a `?expertId=` deep link may still swap this view for the
   * expert's latest thread — the composer stays locked so a draft can't be
   * lost to that navigation. */
  isAdoptingExpertSession?: boolean;
  /** True until a newly hired expert's first kickoff has been handed off. */
  isKickoffStarting?: boolean;
  /** Follow a turn the server started, e.g. after an approval card is answered. */
  onBackendTurn?: () => void;
  /** Which layout floats its controls over the chat's top-left corner on
   *  small viewports; the thread header lines up with them. */
  floatingControls?: FloatingControls;
}

const NO_OP_SEND = () => undefined;
const CONTINUE_AFTER_HELD_CALL = "Continue from where you left off.";

export const ChatContainer = ({
  messages,
  status,
  error,
  sessionId,
  sessionChatStatus,
  sessionSentFrom,
  sessionAutopilotMode = null,
  isLoadingSession,
  isSessionError,
  isSessionNotFound,
  isCreatingSession,
  isReconnecting,
  isFinishProbing,
  isRestoringActiveSession,
  restoreStatusMessage,
  activeStreamStartedAt,
  isUserStopping,
  onCreateSession,
  onSend,
  onStop,
  onEnqueue,
  queuedMessages,
  isUploadingFiles,
  pendingSend,
  hasMoreMessages,
  isLoadingMore,
  onLoadMore,
  droppedFiles,
  onDroppedFilesConsumed,
  turnStats,
  expertIdentity,
  isUsageRefused = false,
  usageFailure,
  onDismissUsage,
  usageRefresh,
  usage = null,
  chatAuthProvider = null,
  isResolvingExpertIdentity,
  isAdoptingExpertSession,
  isKickoffStarting,
  onBackendTurn,
  floatingControls,
}: ChatContainerProps) => {
  const isTaskBarEnabled = useGetFlag(Flag.TASK_PROGRESS_BAR);
  useAutoOpenArtifacts({
    sessionId,
    messages,
    isLoadingSession,
  });
  // isStreaming controls the stop-button UI and routes submits to the queue
  // endpoint — the input itself must NOT be disabled during streaming so users
  // can type and queue their next message. ``isUserStopping`` force-flips
  // this to ``false`` the instant the user clicks Stop, so the button flips
  // to Send regardless of whether AI SDK's status transition has landed yet.
  const isStreaming =
    !isUserStopping && (status === "streaming" || status === "submitted");
  // The input is only truly disabled when the session isn't ready at all
  // (reconnecting, loading, or errored) — NOT during normal streaming and NOT
  // during the wake re-sync (background refetch after long visibility-hidden;
  // locking the input there caused users to see a "stuck" disabled input on
  // refocus when the refetch was slow or revealed stale ``active_stream``
  // state from the backend).
  const isSessionUnavailable =
    !!isReconnecting || isLoadingSession || !!isSessionError;
  const isUsageExhausted = useIsUsageLimitReached(chatAuthProvider);
  const isLimitReached = isUsageExhausted || isUsageRefused;
  const [isUsageTooltipOpen, setIsUsageTooltipOpen] = useState(false);
  const isInputDisabled =
    isSessionUnavailable || !!isResolvingExpertIdentity || !!isKickoffStarting;
  // A fired (archived) expert's threads stay as read-only history — the
  // composer is replaced by a quiet notice so no new turns can be sent.
  const archivedExpertIdentity = expertIdentity?.isArchived
    ? expertIdentity
    : null;
  const isExpertArchived = archivedExpertIdentity !== null;
  const chatExpert =
    !isResolvingExpertIdentity && expertIdentity && !expertIdentity.isArchived
      ? { id: expertIdentity.id, name: expertIdentity.name }
      : null;
  useShareChatExpert(chatExpert);
  const isTurnLocked = isExpertArchived || !!isResolvingExpertIdentity;
  const isSendLocked = isTurnLocked || isLimitReached;
  // The usage limit pauses the composer only. A card's answers, a review
  // approval, a retry or a voice turn still go to the server, which refuses
  // them with the limit dialog and keeps the text for the composer; a no-op
  // here dropped what the user had entered.
  // NO_OP is module-level so a locked composer keeps a stable function identity
  // across renders — otherwise every consumer of `guardedOnSend` (the actions
  // provider, ChatInput, EmptySession, handleRetry) re-renders on each pass.
  const guardedOnSend = isTurnLocked ? NO_OP_SEND : onSend;
  const composer = useComposerBudget({
    sessionId,
    authProvider: chatAuthProvider,
    budget: usage?.budget,
    isLimitReached,
    isOtherwiseLocked: isTurnLocked,
    onSend,
    onCreateSession,
    inputId: SESSION_COMPOSER_ID,
  });
  // While a turn still runs, the limit keeps follow-ups from queueing.
  const atReset = isStreaming ? undefined : composer.atReset;
  // Send at reset takes the paused tooltip's place. Radix reports no close
  // for a tooltip held shut, so an open recorded then would show it, unhovered,
  // once a turn starts: record none, and forget one Send at reset cut short.
  const canShowUsageTooltip = Boolean(isLimitReached && !atReset);
  useEffect(() => {
    if (!canShowUsageTooltip) setIsUsageTooltipOpen(false);
  }, [canShowUsageTooltip]);

  const isVoiceModeEnabled = useGetFlag(Flag.COPILOT_VOICE_MODE);
  const isAutoModeEnabled = useGetFlag(Flag.COPILOT_AUTO_MODE);
  // A thread keeps the mode stored on it; a new expert chat starts on the
  // expert's own default until its first send creates the session.
  const initialMode = sessionId
    ? sessionAutopilotMode
    : (expertIdentity?.autopilotMode ?? null);
  const modeSelector = isAutoModeEnabled ? (
    <AutopilotModeSelector sessionId={sessionId} persistedMode={initialMode} />
  ) : undefined;
  const silenceTimeoutMs = useVoiceSilenceTimeout();
  const voice = useVoiceMode({
    enabled: isVoiceModeEnabled,
    messages,
    isStreaming,
    isReconnecting,
    isFinishProbing,
    sessionId,
    silenceTimeoutMs,
    onSend: guardedOnSend,
  });

  // Measure the usage-limit overlay so the messages scroll area can pad its
  // bottom — otherwise the last message would sit permanently behind the
  // translucent card. Height varies with the card's own content (tier badge,
  // insufficient-credits state, usage-bar layout) so a fixed value would
  // either waste space or clip.
  const usageCardRef = useRef<HTMLDivElement>(null);
  const isMobileLayout = useIsMobile();
  const hasComposerTab = floatingControls === "newLayout";
  // The composer tab overlaps the list's last 41px on small screens, so the
  // list keeps that much more room at the bottom.
  const composerTabInset = hasComposerTab && isMobileLayout ? 41 : 0;
  const [usageCardHeight, setUsageCardHeight] = useState(0);
  useEffect(() => {
    if (!isLimitReached) {
      setUsageCardHeight(0);
      return;
    }
    const el = usageCardRef.current;
    if (!el) return;
    const ro = new ResizeObserver((entries) => {
      const height = entries[0]?.contentRect.height ?? 0;
      setUsageCardHeight(height);
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, [isLimitReached]);

  // Token devtool: estimate the context from loaded history so the badge
  // shows a value before the first live turn. Guarded by breakdownCacheKey so
  // it does not run per stream delta — serializing every part is not free.
  const devtoolBreakdownKeyRef = useRef("");
  useEffect(() => {
    if (!sessionId || !isTokenDevtoolEnabled() || messages.length === 0) return;
    const key = breakdownCacheKey(sessionId, messages, isStreaming);
    if (key === devtoolBreakdownKeyRef.current) return;
    devtoolBreakdownKeyRef.current = key;
    updateHistoryBreakdown(sessionId, messages);
  }, [sessionId, messages, isStreaming]);

  // A chat has to exist before voice mode can send into it, and creating one
  // re-keys this whole subtree — so ask for voice mode and let the new mount
  // start it. The unlock has to happen here, in the click.
  async function handleStartVoiceInNewChat() {
    unlockAudio();
    requestVoiceStart();
    try {
      await onCreateSession();
    } catch (error) {
      takeVoiceStart();
      toastSendFailure({
        title: "Could not start a new chat session",
        error,
        recovery: "voice mode did not start",
      });
    }
  }

  // Retry: re-send the last user message (used by ErrorCard on transient errors).
  const isRetryingRef = useRef(false);
  const handleRetry = useCallback(async () => {
    if (isRetryingRef.current) return;
    isRetryingRef.current = true;
    try {
      const lastRow = [...messages].reverse().find((m) => m.role === "user");
      // A turn an answered card started failed after the call ran: resuming it
      // must not re-send the request that led to the call.
      if (lastRow && isHeldCallRow(lastRow)) {
        await guardedOnSend(CONTINUE_AFTER_HELD_CALL);
        return;
      }
      const lastUserMsg = lastRow;
      const lastText = lastUserMsg?.parts
        .filter(
          (p): p is Extract<typeof p, { type: "text" }> => p.type === "text",
        )
        .map((p) => p.text)
        .join("");
      if (lastText) {
        const kickoffExpertId = lastUserMsg
          ? getKickoffExpertId(lastUserMsg)
          : null;
        const kickoffAttemptToken = lastUserMsg
          ? getKickoffAttemptToken(lastUserMsg)
          : null;
        await guardedOnSend(
          kickoffExpertId ? stripLegacyKickoffMarker(lastText) : lastText,
          undefined,
          undefined,
          kickoffExpertId
            ? {
                kind: "expert_kickoff",
                expertId: kickoffExpertId,
                ...(kickoffAttemptToken
                  ? { attemptToken: kickoffAttemptToken }
                  : {}),
              }
            : undefined,
        );
      }
    } catch (error) {
      toastSendFailure({
        title: "Couldn't retry message",
        error,
        recovery: "your previous message is still in the chat",
      });
    } finally {
      isRetryingRef.current = false;
    }
  }, [guardedOnSend, messages]);

  return (
    <CopilotChatActionsProvider
      onSend={guardedOnSend}
      onBackendTurn={onBackendTurn}
      sessionId={sessionId}
      assistantName={expertIdentity?.name ?? "Otto"}
    >
      <LayoutGroup id="copilot-2-chat-layout">
        <div className="flex h-full min-h-0 w-full flex-col px-2 lg:px-0">
          {/* The chat column runs full width: the max-w-3xl cap lives on the
                message list and the input instead, so the expert thread header
                can span edge to edge while staying aligned with the messages. */}
          {sessionId && isSessionNotFound ? (
            <SessionNotFound />
          ) : sessionId ? (
            <div className="relative flex h-full min-h-0 w-full flex-col bg-white">
              <div className="absolute right-0 top-0 z-30">
                <ContextPanelToggle
                  sessionId={sessionId}
                  expert={chatExpert}
                  hideIntegrationsOnMobile={floatingControls === "newLayout"}
                />
              </div>
              <NewChatOnboarding
                expertId={expertIdentity?.id ?? null}
                enabled={
                  messages.length === 0 &&
                  !isInputDisabled &&
                  !isLimitReached &&
                  !isStreaming &&
                  !isCreatingSession &&
                  !isExpertArchived
                }
              >
                <ChatMessagesContainer
                  messages={messages}
                  status={status}
                  error={error}
                  isLoading={isLoadingSession}
                  isRestoringActiveSession={isRestoringActiveSession}
                  restoreStatusMessage={restoreStatusMessage}
                  activeStreamStartedAt={activeStreamStartedAt}
                  sessionID={sessionId}
                  sessionChatStatus={sessionChatStatus}
                  sessionSentFrom={sessionSentFrom}
                  hasMoreMessages={hasMoreMessages}
                  isLoadingMore={isLoadingMore}
                  onLoadMore={onLoadMore}
                  onRetry={handleRetry}
                  turnStats={turnStats}
                  queuedMessages={queuedMessages}
                  pendingSend={pendingSend}
                  bottomContentPadding={usageCardHeight + composerTabInset}
                  expertIdentity={expertIdentity}
                  isResolvingExpertIdentity={isResolvingExpertIdentity}
                  floatingControls={floatingControls}
                />
              </NewChatOnboarding>
              {archivedExpertIdentity ? (
                <>
                  {hasComposerTab && (
                    <div className="mx-auto w-full max-w-3xl px-3">
                      <ComposerPanelTab expert={null} />
                    </div>
                  )}
                  <ArchivedExpertNotice
                    expertName={archivedExpertIdentity.name}
                    reason={archivedExpertIdentity.readOnlyReason ?? "fired"}
                  />
                </>
              ) : (
                <ChatComposerFrame
                  className={cn(hasComposerTab && "max-lg:pt-0")}
                  paused={isLimitReached && !atReset}
                  noticeRef={usageCardRef}
                  notice={
                    isLimitReached ? (
                      <UsageLimitReachedCard
                        sessionID={sessionId}
                        failure={usageFailure}
                        onDismiss={onDismissUsage}
                        refresh={usageRefresh}
                      />
                    ) : null
                  }
                >
                  <SharedChatNotice sessionId={sessionId} />
                  {hasComposerTab && <ComposerPanelTab expert={chatExpert} />}
                  {isTaskBarEnabled && (
                    <div className="relative z-10">
                      <TaskProgressBar
                        todos={getLatestTaskList(messages) ?? []}
                        isStreaming={isStreaming}
                      />
                    </div>
                  )}
                  {composer.scheduled.pending && (
                    <ScheduledSendChip
                      job={composer.scheduled.pending}
                      onEdit={() => void composer.scheduled.edit()}
                      onCancel={() => void composer.scheduled.cancel()}
                      isBusy={composer.scheduled.isBusy}
                    />
                  )}
                  <Tooltip
                    open={canShowUsageTooltip && isUsageTooltipOpen}
                    onOpenChange={(open) =>
                      setIsUsageTooltipOpen(open && canShowUsageTooltip)
                    }
                  >
                    <TooltipTrigger asChild>
                      <div>
                        <ChatInput
                          inputId={SESSION_COMPOSER_ID}
                          draftKey={sessionDraftKey(sessionId)}
                          onSend={guardedOnSend}
                          disabled={isInputDisabled}
                          sendDisabled={isLimitReached && !atReset}
                          atReset={atReset}
                          isStreaming={isStreaming}
                          isTurnStarting={status === "submitted"}
                          isUploadingFiles={isUploadingFiles}
                          onStop={onStop}
                          onEnqueue={onEnqueue}
                          placeholder={
                            (atReset && composer.placeholder) ||
                            "What else can I help with?"
                          }
                          droppedFiles={droppedFiles}
                          onDroppedFilesConsumed={onDroppedFilesConsumed}
                          hasSession={!!sessionId}
                          sessionId={sessionId}
                          expertId={expertIdentity?.id ?? null}
                          modeSelector={modeSelector}
                          expertName={expertIdentity?.name ?? null}
                          voiceMode={
                            isVoiceModeEnabled
                              ? {
                                  onStart: voice.toggle,
                                  disabled:
                                    isInputDisabled ||
                                    isSendLocked ||
                                    voice.isStarting,
                                }
                              : undefined
                          }
                          voiceBar={
                            voice.isActive ? (
                              <VoiceModeBar
                                state={voice.state}
                                statusLabel={voice.statusLabel}
                                failure={voice.failure}
                                onRetry={voice.retryFailedUtterance}
                                onDownload={voice.downloadFailedUtterance}
                                leaveButton={
                                  <VoiceModeButton
                                    isActive
                                    speaking={voice.state === "speaking"}
                                    onClick={voice.toggle}
                                  />
                                }
                              />
                            ) : undefined
                          }
                        />
                      </div>
                    </TooltipTrigger>
                    <TooltipContent side="top" className="max-w-sm">
                      Sending is paused. Your draft is saved and you can keep
                      editing it.
                    </TooltipContent>
                  </Tooltip>
                  {composer.showLowHint && usage && (
                    <BudgetLowHint usage={usage} />
                  )}
                </ChatComposerFrame>
              )}
            </div>
          ) : (
            <EmptySession
              isCreatingSession={isCreatingSession}
              onCreateSession={onCreateSession}
              onSend={guardedOnSend}
              modeSelector={modeSelector}
              voiceMode={
                isVoiceModeEnabled
                  ? {
                      onStart: handleStartVoiceInNewChat,
                      disabled:
                        isInputDisabled || isSendLocked || isCreatingSession,
                    }
                  : undefined
              }
              isUploadingFiles={isUploadingFiles}
              droppedFiles={droppedFiles}
              onDroppedFilesConsumed={onDroppedFilesConsumed}
              isInteractionLocked={
                (isSendLocked && !composer.atReset) || !!isAdoptingExpertSession
              }
              atReset={composer.atReset}
              usageNotice={
                isLimitReached ? (
                  <UsageLimitReachedCard refresh={usageRefresh} />
                ) : undefined
              }
              lowHint={
                composer.showLowHint && usage ? (
                  <BudgetLowHint usage={usage} />
                ) : undefined
              }
              isKickoffStarting={isKickoffStarting}
              expertName={expertIdentity?.name}
              expertId={expertIdentity?.id ?? null}
            />
          )}
        </div>
      </LayoutGroup>
    </CopilotChatActionsProvider>
  );
};
