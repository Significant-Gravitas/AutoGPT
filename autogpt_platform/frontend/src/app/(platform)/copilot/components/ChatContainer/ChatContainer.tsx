"use client";
import { ChatInput } from "@/app/(platform)/copilot/components/ChatInput/ChatInput";
import { cn } from "@/lib/utils";
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";
import { UIDataTypes, UIMessage, UITools } from "ai";
import { LayoutGroup, motion } from "framer-motion";
import { useCallback } from "react";
import { useCopilotUIStore } from "../../store";
import { ChatMessagesContainer } from "../ChatMessagesContainer/ChatMessagesContainer";
import { CopilotChatActionsProvider } from "../CopilotChatActionsProvider/CopilotChatActionsProvider";
import { EmptySession } from "../EmptySession/EmptySession";
import { useAutoOpenArtifacts } from "./useAutoOpenArtifacts";

export interface ChatContainerProps {
  messages: UIMessage<unknown, UIDataTypes, UITools>[];
  status: string;
  error: Error | undefined;
  sessionId: string | null;
  isLoadingSession: boolean;
  isSessionError?: boolean;
  isCreatingSession: boolean;
  /** True when backend has an active stream but we haven't reconnected yet. */
  isReconnecting?: boolean;
  /** True while re-syncing session state after device wake. */
  isSyncing?: boolean;
  onCreateSession: () => void | Promise<string>;
  onSend: (message: string, files?: File[]) => void | Promise<void>;
  onStop: () => void;
  /** Called to enqueue a message while streaming (bypasses normal send flow). */
  onEnqueue?: (message: string) => void | Promise<void>;
  /** A pending queued message waiting to be injected, shown at the end of chat. */
  queuedMessage?: string | null;
  isUploadingFiles?: boolean;
  hasMoreMessages?: boolean;
  isLoadingMore?: boolean;
  onLoadMore?: () => void;
  /** Files dropped onto the chat window. */
  droppedFiles?: File[];
  /** Called after droppedFiles have been consumed by ChatInput. */
  onDroppedFilesConsumed?: () => void;
  /** Duration in ms for historical turns, keyed by message ID. */
  historicalDurations?: Map<string, number>;
}
export const ChatContainer = ({
  messages,
  status,
  error,
  sessionId,
  isLoadingSession,
  isSessionError,
  isCreatingSession,
  isReconnecting,
  isSyncing,
  onCreateSession,
  onSend,
  onStop,
  onEnqueue,
  queuedMessage,
  isUploadingFiles,
  hasMoreMessages,
  isLoadingMore,
  onLoadMore,
  droppedFiles,
  onDroppedFilesConsumed,
  historicalDurations,
}: ChatContainerProps) => {
  const isArtifactsEnabled = useGetFlag(Flag.ARTIFACTS);
  const isArtifactPanelOpen = useCopilotUIStore((s) => s.artifactPanel.isOpen);
  // When the flag is off we must not auto-open artifacts or let the panel's
  // open state drive layout width; an artifact generated in a stale session
  // state would otherwise shrink the chat column with no panel rendered.
  const isArtifactOpen = isArtifactsEnabled && isArtifactPanelOpen;
  useAutoOpenArtifacts({
    messages: isArtifactsEnabled ? messages : [],
    sessionId,
  });
  // isStreaming controls the stop-button UI and routes submits to the queue
  // endpoint — the input itself must NOT be disabled during streaming so users
  // can type and queue their next message.
  const isStreaming = status === "streaming" || status === "submitted";
  // The input is only truly disabled when the session isn't ready at all
  // (reconnecting, syncing, loading, or errored) — NOT during normal streaming.
  const isInputDisabled =
    !!isReconnecting || !!isSyncing || isLoadingSession || !!isSessionError;
  const inputLayoutId = "copilot-2-chat-input";

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
        <div className="flex h-full min-h-0 w-full flex-col bg-[#f8f8f9] px-2 lg:px-0">
          {sessionId ? (
            <div
              className={cn(
                "mx-auto flex h-full min-h-0 w-full flex-col",
                !isArtifactOpen && "max-w-3xl",
              )}
            >
              <ChatMessagesContainer
                messages={messages}
                status={status}
                error={error}
                isLoading={isLoadingSession}
                sessionID={sessionId}
                hasMoreMessages={hasMoreMessages}
                isLoadingMore={isLoadingMore}
                onLoadMore={onLoadMore}
                onRetry={handleRetry}
                historicalDurations={historicalDurations}
                queuedMessage={queuedMessage}
              />
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ duration: 0.3 }}
                className="relative px-3 pb-2 pt-2"
              >
                <div className="pointer-events-none absolute left-0 right-0 top-[-18px] z-10 h-6 bg-gradient-to-b from-transparent to-[#f8f8f9]" />
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
              </motion.div>
            </div>
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
