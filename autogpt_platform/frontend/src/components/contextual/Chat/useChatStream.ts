"use client";

import { useEffect, useRef, useState } from "react";
import { toast } from "sonner";
import { useChatStore } from "./chat-store";
import type { StreamChunk } from "./chat-types";

export type { StreamChunk } from "./chat-types";

export function useChatStream() {
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const currentSessionIdRef = useRef<string | null>(null);
  const store = useChatStore();
  const onChunkCallbackRef = useRef<((chunk: StreamChunk) => void) | null>(
    null,
  );

  function stopStreaming(sessionId?: string) {
    const targetSession = sessionId || currentSessionIdRef.current;
    if (targetSession) {
      store.stopStream(targetSession);
      store.unregisterActiveSession(targetSession);
    }
    setIsStreaming(false);
  }

  useEffect(
    function cleanupOnUnmount() {
      return function cleanup() {
        const sessionId = currentSessionIdRef.current;
        if (sessionId) {
          const isActive = store.isSessionActive(sessionId);
          if (!isActive) {
            store.stopStream(sessionId);
          }
        }
        currentSessionIdRef.current = null;
        onChunkCallbackRef.current = null;
      };
    },
    [store],
  );

  useEffect(
    function syncStreamingState() {
      const sessionId = currentSessionIdRef.current;
      if (!sessionId) return;

      const status = store.getStreamStatus(sessionId);
      const shouldBeStreaming = status === "streaming";
      if (shouldBeStreaming !== isStreaming) setIsStreaming(shouldBeStreaming);

      if (status === "error") {
        const completed = store.getCompletedStream(sessionId);
        if (completed?.error) setError(completed.error);
      }
    },
    [store, isStreaming],
  );

  async function sendMessage(
    sessionId: string,
    message: string,
    onChunk: (chunk: StreamChunk) => void,
    isUserMessage: boolean = true,
    context?: { url: string; content: string },
  ) {
    const previousSessionId = currentSessionIdRef.current;
    if (previousSessionId && previousSessionId !== sessionId) {
      store.stopStream(previousSessionId);
    }

    currentSessionIdRef.current = sessionId;
    onChunkCallbackRef.current = onChunk;
    setIsStreaming(true);
    setError(null);

    store.registerActiveSession(sessionId);

    try {
      await store.startStream(
        sessionId,
        message,
        isUserMessage,
        context,
        onChunk,
      );

      const status = store.getStreamStatus(sessionId);
      if (status === "error") {
        const completed = store.getCompletedStream(sessionId);
        if (completed?.error) {
          setError(completed.error);
          toast.error("Connection Failed", {
            description: "Unable to connect to chat service. Please try again.",
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
  }

  return {
    isStreaming,
    error,
    sendMessage,
    stopStreaming,
  };
}
