"use client";

import { useChatStreamStore } from "./chat-stream-store";
import type { StreamChunk } from "./stream-utils";
import { useCallback, useEffect, useRef, useState } from "react";

interface UseManagedChatStreamArgs {
  sessionId: string | null;
  onChunk?: (chunk: StreamChunk) => void;
}

export function useManagedChatStream({
  sessionId,
  onChunk,
}: UseManagedChatStreamArgs) {
  const store = useChatStreamStore();
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const onChunkRef = useRef(onChunk);
  onChunkRef.current = onChunk;

  useEffect(
    function subscribeToExistingStream() {
      if (!sessionId) return;

      function handleChunk(chunk: StreamChunk) {
        if (chunk.type === "stream_end") {
          setIsStreaming(false);
        } else if (chunk.type === "error") {
          setIsStreaming(false);
          setError(new Error(chunk.message || "Stream error"));
        }
        onChunkRef.current?.(chunk);
      }

      const streamStatus = store.getStreamStatus(sessionId);
      setIsStreaming(streamStatus === "streaming");

      const unsubscribe = store.subscribeToStream(sessionId, handleChunk);

      return unsubscribe;
    },
    [store, sessionId],
  );

  const sendMessage = useCallback(
    async function sendMessage(
      message: string,
      isUserMessage: boolean = true,
      context?: { url: string; content: string },
    ) {
      if (!sessionId) {
        console.error(
          "[useManagedChatStream] Cannot send message: no session ID",
        );
        return;
      }

      setIsStreaming(true);
      setError(null);

      function handleChunk(chunk: StreamChunk) {
        if (chunk.type === "stream_end") {
          setIsStreaming(false);
        } else if (chunk.type === "error") {
          setIsStreaming(false);
          setError(new Error(chunk.message || "Stream error"));
        }
        onChunkRef.current?.(chunk);
      }

      try {
        await store.startStream(
          sessionId,
          message,
          isUserMessage,
          context,
          handleChunk,
        );
      } catch (err) {
        setError(err instanceof Error ? err : new Error("Failed to send"));
        setIsStreaming(false);
      }
    },
    [store, sessionId],
  );

  const stopStreaming = useCallback(
    function stopStreaming() {
      if (!sessionId) return;
      store.stopStream(sessionId);
      setIsStreaming(false);
    },
    [store, sessionId],
  );

  return {
    isStreaming,
    error,
    sendMessage,
    stopStreaming,
  };
}
