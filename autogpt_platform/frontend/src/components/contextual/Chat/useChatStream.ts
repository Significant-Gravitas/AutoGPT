import { useChatStreamStore } from "@/providers/chat-stream/chat-stream-store";
import type { StreamChunk } from "@/providers/chat-stream/stream-utils";
import { useCallback, useEffect, useRef, useState } from "react";
import { toast } from "sonner";

export type { StreamChunk } from "@/providers/chat-stream/stream-utils";

export function useChatStream() {
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const currentSessionIdRef = useRef<string | null>(null);
  const streamManager = useChatStreamStore();
  const onChunkCallbackRef = useRef<((chunk: StreamChunk) => void) | null>(
    null,
  );

  const stopStreaming = useCallback(
    function stopStreaming(sessionId?: string) {
      const targetSession = sessionId || currentSessionIdRef.current;
      if (targetSession) {
        streamManager.stopStream(targetSession);
        streamManager.unregisterActiveSession(targetSession);
      }
      setIsStreaming(false);
    },
    [streamManager],
  );

  useEffect(function cleanupOnUnmount() {
    return function cleanup() {
      const sessionId = currentSessionIdRef.current;
      if (sessionId) {
        const isActive = streamManager.isSessionActive(sessionId);
        if (!isActive) {
          streamManager.stopStream(sessionId);
        }
      }
      currentSessionIdRef.current = null;
      onChunkCallbackRef.current = null;
    };
  }, [streamManager]);

  useEffect(
    function syncStreamingState() {
      const sessionId = currentSessionIdRef.current;
      if (!sessionId) return;

      const status = streamManager.getStreamStatus(sessionId);
      const shouldBeStreaming = status === "streaming";
      if (shouldBeStreaming !== isStreaming) setIsStreaming(shouldBeStreaming);

      if (status === "error") {
        const completed = streamManager.getCompletedStream(sessionId);
        if (completed?.error) setError(completed.error);
      }
    },
    [streamManager, isStreaming],
  );

  const sendMessage = useCallback(
    async function sendMessage(
      sessionId: string,
      message: string,
      onChunk: (chunk: StreamChunk) => void,
      isUserMessage: boolean = true,
      context?: { url: string; content: string },
    ) {
      const previousSessionId = currentSessionIdRef.current;
      if (previousSessionId && previousSessionId !== sessionId) {
        streamManager.stopStream(previousSessionId);
      }

      currentSessionIdRef.current = sessionId;
      onChunkCallbackRef.current = onChunk;
      setIsStreaming(true);
      setError(null);

      streamManager.registerActiveSession(sessionId);

      try {
        await streamManager.startStream(
          sessionId,
          message,
          isUserMessage,
          context,
          onChunk,
        );

        const status = streamManager.getStreamStatus(sessionId);
        if (status === "error") {
          const completed = streamManager.getCompletedStream(sessionId);
          if (completed?.error) {
            setError(completed.error);
            toast.error("Connection Failed", {
              description:
                "Unable to connect to chat service. Please try again.",
            });
            throw completed.error;
          }
        }
      } catch (err) {
        const streamError =
          err instanceof Error ? err : new Error("Failed to start stream");
        setError(streamError);
        throw streamError;
      } finally {
        setIsStreaming(false);
      }
    },
    [streamManager],
  );

  return {
    isStreaming,
    error,
    sendMessage,
    stopStreaming,
  };
}
