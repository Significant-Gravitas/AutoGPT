"use client";

import { useCopilotSessionId } from "@/app/(platform)/copilot/useCopilotSessionId";
import { LoadingSpinner } from "@/components/atoms/LoadingSpinner/LoadingSpinner";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import { useEffect, useRef } from "react";
import { ChatContainer } from "./components/ChatContainer/ChatContainer";
import { ChatErrorState } from "./components/ChatErrorState/ChatErrorState";
import { useChat } from "./useChat";

export interface ChatProps {
  className?: string;
  initialPrompt?: string;
  onSessionNotFound?: () => void;
  onStreamingChange?: (isStreaming: boolean) => void;
}

export function Chat({
  className,
  initialPrompt,
  onSessionNotFound,
  onStreamingChange,
}: ChatProps) {
  const { urlSessionId } = useCopilotSessionId();
  const hasHandledNotFoundRef = useRef(false);
  const {
    session,
    messages,
    isLoading,
    isCreating,
    error,
    isSessionNotFound,
    sessionId,
    createSession,
    showLoader,
    startPollingForOperation,
  } = useChat({ urlSessionId });

  // Extract active stream info for reconnection
  const activeStream = (
    session as {
      active_stream?: {
        task_id: string;
        last_message_id: string;
        operation_id: string;
        tool_name: string;
      };
    }
  )?.active_stream;

  useEffect(() => {
    if (!onSessionNotFound) return;
    if (!urlSessionId) return;
    if (!isSessionNotFound || isLoading || isCreating) return;
    if (hasHandledNotFoundRef.current) return;
    hasHandledNotFoundRef.current = true;
    onSessionNotFound();
  }, [
    onSessionNotFound,
    urlSessionId,
    isSessionNotFound,
    isLoading,
    isCreating,
  ]);

  const shouldShowLoader = showLoader && (isLoading || isCreating);

  return (
    <div className={cn("flex h-full flex-col", className)}>
      {/* Main Content */}
      <main className="flex min-h-0 w-full flex-1 flex-col overflow-hidden bg-[#f8f8f9]">
        {/* Loading State */}
        {shouldShowLoader && (
          <div className="flex flex-1 items-center justify-center">
            <div className="flex flex-col items-center gap-3">
              <LoadingSpinner size="large" className="text-neutral-400" />
              <Text variant="body" className="text-zinc-500">
                Loading your chat...
              </Text>
            </div>
          </div>
        )}

        {/* Error State */}
        {error && !isLoading && (
          <ChatErrorState error={error} onRetry={createSession} />
        )}

        {/* Session Content */}
        {sessionId && !isLoading && !error && (
          <ChatContainer
            sessionId={sessionId}
            initialMessages={messages}
            initialPrompt={initialPrompt}
            className="flex-1"
            onStreamingChange={onStreamingChange}
            onOperationStarted={startPollingForOperation}
            activeStream={
              activeStream
                ? {
                    taskId: activeStream.task_id,
                    lastMessageId: activeStream.last_message_id,
                    operationId: activeStream.operation_id,
                    toolName: activeStream.tool_name,
                  }
                : undefined
            }
          />
        )}
      </main>
    </div>
  );
}
