"use client";

import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import { useEffect, useRef } from "react";
import { ChatContainer } from "./components/ChatContainer/ChatContainer";
import { ChatErrorState } from "./components/ChatErrorState/ChatErrorState";
import { ChatLoader } from "./components/ChatLoader/ChatLoader";
import { useChat } from "./useChat";

export interface ChatProps {
  className?: string;
  urlSessionId?: string | null;
  initialPrompt?: string;
  onSessionNotFound?: () => void;
  onStreamingChange?: (isStreaming: boolean) => void;
}

export function Chat({
  className,
  urlSessionId,
  initialPrompt,
  onSessionNotFound,
  onStreamingChange,
}: ChatProps) {
  const hasHandledNotFoundRef = useRef(false);
  const {
    messages,
    isLoading,
    isCreating,
    error,
    isSessionNotFound,
    sessionId,
    createSession,
    showLoader,
  } = useChat({ urlSessionId });

  useEffect(
    function handleMissingSession() {
      if (!onSessionNotFound) return;
      if (!urlSessionId) return;
      if (!isSessionNotFound || isLoading || isCreating) return;
      if (hasHandledNotFoundRef.current) return;
      hasHandledNotFoundRef.current = true;
      onSessionNotFound();
    },
    [onSessionNotFound, urlSessionId, isSessionNotFound, isLoading, isCreating],
  );

  return (
    <div className={cn("flex h-full flex-col", className)}>
      {/* Main Content */}
      <main className="flex min-h-0 w-full flex-1 flex-col overflow-hidden bg-[#f8f8f9]">
        {/* Loading State */}
        {showLoader && (isLoading || isCreating) && (
          <div className="flex flex-1 items-center justify-center">
            <div className="flex flex-col items-center gap-4">
              <ChatLoader />
              <Text variant="body" className="text-zinc-500">
                Loading your chats...
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
          />
        )}
      </main>
    </div>
  );
}
