"use client";
import { ChatInput } from "@/app/(platform)/copilot/components/ChatInput/ChatInput";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/atoms/Tooltip/BaseTooltip";
import { cn } from "@/lib/utils";
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";
import { UIDataTypes, UIMessage, UITools } from "ai";
import { LayoutGroup, motion } from "framer-motion";
import { useCallback, useEffect, useRef, useState } from "react";
import type { TurnStatsMap } from "../../helpers/convertChatSessionToUiMessages";
import { ChatMessagesContainer } from "../ChatMessagesContainer/ChatMessagesContainer";
import { CopilotChatActionsProvider } from "../CopilotChatActionsProvider/CopilotChatActionsProvider";
import { EmptySession } from "../EmptySession/EmptySession";
import { UsageLimitReachedCard } from "../UsageLimits/UsageLimitReachedCard/UsageLimitReachedCard";
import { useIsUsageLimitReached } from "../UsageLimits/useIsUsageLimitReached";
import { SharedChatNotice } from "./components/SharedChatNotice";
import { useAutoOpenArtifacts } from "./useAutoOpenArtifacts";

export interface ChatContainerProps {
  messages: UIMessage<unknown, UIDataTypes, UITools>[];
  status: string;
  error: Error | undefined;
  sessionId: string | null;
  sessionChatStatus?: string;
  isLoadingSession: boolean;
  isSessionError?: boolean;
  isCreatingSession: boolean;
  /** True when backend has an active stream but we haven't reconnected yet. */
  isReconnecting?: boolean;
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
  onSend: (message: string, files?: File[]) => void | Promise<void>;
  onStop: () => void;
  /** Called to enqueue a message while streaming (bypasses normal send flow). */
  onEnqueue?: (message: string) => void | Promise<void>;
  /** Pending queued messages waiting to be injected, shown at the end of chat. */
  queuedMessages?: string[];
  isUploadingFiles?: boolean;
  hasMoreMessages?: boolean;
  isLoadingMore?: boolean;
  onLoadMore?: () => void;
  /** Files dropped onto the chat window. */
  droppedFiles?: File[];
  /** Called after droppedFiles have been consumed by ChatInput. */
  onDroppedFilesConsumed?: () => void;
  /** Per-message stats (durationMs, createdAt), keyed by message ID. */
  turnStats?: TurnStatsMap;
}
export const ChatContainer = ({
  messages,
  status,
  error,
  sessionId,
  sessionChatStatus,
  isLoadingSession,
  isSessionError,
  isCreatingSession,
  isReconnecting,
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
  hasMoreMessages,
  isLoadingMore,
  onLoadMore,
  droppedFiles,
  onDroppedFilesConsumed,
  turnStats,
}: ChatContainerProps) => {
  const isArtifactsEnabled = useGetFlag(Flag.ARTIFACTS);
  useAutoOpenArtifacts({
    sessionId,
    messages,
    isLoadingSession,
    isArtifactsEnabled,
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
  const isLimitReached = useIsUsageLimitReached();
  const isInputDisabled = isSessionUnavailable || isLimitReached;
  const inputLayoutId = "copilot-2-chat-input";

  // Measure the usage-limit overlay so the messages scroll area can pad its
  // bottom — otherwise the last message would sit permanently behind the
  // translucent card. Height varies with the card's own content (tier badge,
  // insufficient-credits state, usage-bar layout) so a fixed value would
  // either waste space or clip.
  const usageCardRef = useRef<HTMLDivElement>(null);
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

  // Retry: re-send the last user message (used by ErrorCard on transient errors)
  const handleRetry = useCallback(() => {
    const lastUserMsg = [...messages].reverse().find((m) => m.role === "user");
    const lastText = lastUserMsg?.parts
      .filter(
        (p): p is Extract<typeof p, { type: "text" }> => p.type === "text",
      )
      .map((p) => p.text)
      .join("");
    if (lastText) {
      onSend(lastText);
    }
  }, [messages, onSend]);

  return (
    <CopilotChatActionsProvider onSend={onSend}>
      <LayoutGroup id="copilot-2-chat-layout">
        <div className="flex h-full min-h-0 flex-col bg-white">
          {sessionId ? (
            <>
              <div className="flex min-h-0 flex-1 flex-col px-2 pt-[15px] lg:px-0">
                <div
                  className={cn(
                    "mx-auto flex h-full min-h-0 w-full flex-col",
                    "max-w-3xl",
                  )}
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
                    hasMoreMessages={hasMoreMessages}
                    isLoadingMore={isLoadingMore}
                    onLoadMore={onLoadMore}
                    onRetry={handleRetry}
                    turnStats={turnStats}
                    queuedMessages={queuedMessages}
                    bottomContentPadding={usageCardHeight}
                  />
                </div>
              </div>
              <div className="mt-4">
                <div className="h-px w-full bg-neutral-200" />
                <div className="h-px w-full bg-white" />
              </div>
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ duration: 0.3 }}
                className="relative bg-zinc-50"
              >
                <div
                  className={cn(
                    "relative mx-auto w-full px-3 pb-2 pt-0",
                    "max-w-3xl",
                  )}
                >
                  {isLimitReached && (
                    <div
                      ref={usageCardRef}
                      className="pointer-events-none absolute bottom-full left-0 right-0 z-20 mb-2.5 pb-2"
                    >
                      <div
                        aria-hidden="true"
                        data-testid="usage-limit-backdrop"
                        className="absolute -inset-x-14 -top-20 bottom-[-18px] overflow-hidden rounded-[2rem] bg-[radial-gradient(ellipse_at_center,rgba(255,255,255,0.96)_0%,rgba(255,255,255,0.9)_42%,rgba(255,255,255,0.58)_68%,rgba(255,255,255,0)_100%)] backdrop-blur-lg [mask-image:linear-gradient(to_bottom,transparent_0%,black_26%,black_100%)]"
                      >
                        <div className="absolute inset-x-10 bottom-0 h-28 rounded-full bg-white/80 blur-2xl" />
                        <div className="absolute inset-x-16 bottom-8 h-16 rounded-full bg-white/55 blur-xl" />
                      </div>
                      <div className="pointer-events-auto relative px-3">
                        <UsageLimitReachedCard />
                      </div>
                    </div>
                  )}
                  <SharedChatNotice sessionId={sessionId} />
                  <Tooltip open={isLimitReached ? undefined : false}>
                    <TooltipTrigger asChild>
                      <div>
                        <ChatInput
                          inputId="chat-input-session"
                          onSend={onSend}
                          disabled={isInputDisabled}
                          isStreaming={isStreaming}
                          isUploadingFiles={isUploadingFiles}
                          onStop={onStop}
                          onEnqueue={onEnqueue}
                          placeholder="What else can I help with?"
                          droppedFiles={droppedFiles}
                          onDroppedFilesConsumed={onDroppedFilesConsumed}
                          hasSession={!!sessionId}
                        />
                      </div>
                    </TooltipTrigger>
                    <TooltipContent side="top" className="max-w-sm">
                      You&apos;ve reached your usage limit. Wait for it to
                      refresh or upgrade your plan to continue sending messages.
                    </TooltipContent>
                  </Tooltip>
                </div>
              </motion.div>
            </>
          ) : (
            <EmptySession
              inputLayoutId={inputLayoutId}
              isCreatingSession={isCreatingSession}
              onCreateSession={onCreateSession}
              onSend={onSend}
              isUploadingFiles={isUploadingFiles}
              droppedFiles={droppedFiles}
              onDroppedFilesConsumed={onDroppedFilesConsumed}
            />
          )}
        </div>
      </LayoutGroup>
    </CopilotChatActionsProvider>
  );
};
