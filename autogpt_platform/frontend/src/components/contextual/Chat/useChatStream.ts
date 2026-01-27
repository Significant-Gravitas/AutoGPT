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
  const onChunkCallbackRef = useRef<((chunk: StreamChunk) => void) | null>(
    null,
  );

  const stopStream = useChatStore((s) => s.stopStream);
  const unregisterActiveSession = useChatStore(
    (s) => s.unregisterActiveSession,
  );
  const isSessionActive = useChatStore((s) => s.isSessionActive);
  const onStreamComplete = useChatStore((s) => s.onStreamComplete);
  const getCompletedStream = useChatStore((s) => s.getCompletedStream);
  const registerActiveSession = useChatStore((s) => s.registerActiveSession);
  const startStream = useChatStore((s) => s.startStream);
  const getStreamStatus = useChatStore((s) => s.getStreamStatus);

  function stopStreaming(sessionId?: string) {
    const targetSession = sessionId || currentSessionIdRef.current;
    if (targetSession) {
      stopStream(targetSession);
      unregisterActiveSession(targetSession);
    }
    setIsStreaming(false);
  }

  useEffect(() => {
    return function cleanup() {
      const sessionId = currentSessionIdRef.current;
      if (sessionId && !isSessionActive(sessionId)) {
        stopStream(sessionId);
      }
      currentSessionIdRef.current = null;
      onChunkCallbackRef.current = null;
    };
  }, []);

  useEffect(() => {
    const unsubscribe = onStreamComplete(
      function handleStreamComplete(completedSessionId) {
        if (completedSessionId !== currentSessionIdRef.current) return;

        setIsStreaming(false);
        const completed = getCompletedStream(completedSessionId);
        if (completed?.error) {
          setError(completed.error);
        }
        unregisterActiveSession(completedSessionId);
      },
    );

    return unsubscribe;
  }, []);

  async function sendMessage(
    sessionId: string,
    message: string,
    onChunk: (chunk: StreamChunk) => void,
    isUserMessage: boolean = true,
    context?: { url: string; content: string },
  ) {
    const previousSessionId = currentSessionIdRef.current;
    if (previousSessionId && previousSessionId !== sessionId) {
      stopStreaming(previousSessionId);
    }

    currentSessionIdRef.current = sessionId;
    onChunkCallbackRef.current = onChunk;
    setIsStreaming(true);
    setError(null);

    registerActiveSession(sessionId);

    try {
      await startStream(sessionId, message, isUserMessage, context, onChunk);

      const status = getStreamStatus(sessionId);
      if (status === "error") {
        const completed = getCompletedStream(sessionId);
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
