import { useState, useCallback, useRef } from "react";
import { toast } from "sonner";
import type { ToolArguments, ToolResult } from "@/types/chat";

const MAX_RETRIES = 3;
const INITIAL_RETRY_DELAY = 1000; // 1 second

export interface StreamChunk {
  type:
    | "text_chunk"
    | "tool_call"
    | "tool_response"
    | "login_needed"
    | "credentials_needed"
    | "error"
    | "usage"
    | "stream_end";
  timestamp?: string;
  content?: string;
  message?: string;
  // Tool call/response fields
  tool_id?: string;
  tool_name?: string;
  arguments?: ToolArguments;
  result?: ToolResult;
  success?: boolean;
  // Login needed fields
  session_id?: string;
  // Credentials needed fields
  provider?: string;
  provider_name?: string;
  credential_type?: string;
  scopes?: string[];
  title?: string;
  [key: string]: unknown; // Allow additional fields
}

interface UseChatStreamResult {
  isStreaming: boolean;
  error: Error | null;
  sendMessage: (
    sessionId: string,
    message: string,
    onChunk: (chunk: StreamChunk) => void,
  ) => Promise<void>;
  stopStreaming: () => void;
}

export function useChatStream(): UseChatStreamResult {
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const eventSourceRef = useRef<EventSource | null>(null);
  const retryCountRef = useRef<number>(0);
  const retryTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  const stopStreaming = useCallback(function stopStreaming() {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
      eventSourceRef.current = null;
    }
    if (retryTimeoutRef.current) {
      clearTimeout(retryTimeoutRef.current);
      retryTimeoutRef.current = null;
    }
    retryCountRef.current = 0;
    setIsStreaming(false);
  }, []);

  const sendMessage = useCallback(
    async function sendMessage(
      sessionId: string,
      message: string,
      onChunk: (chunk: StreamChunk) => void,
    ) {
      // Stop any existing stream
      stopStreaming();

      // Reset retry count for new message
      retryCountRef.current = 0;

      setIsStreaming(true);
      setError(null);

      try {
        // EventSource doesn't support custom headers, so we use a Next.js API route
        // that acts as a proxy and adds authentication headers server-side
        // This matches the pattern from PR #10905 where SSE went through the same server
        const url = `/api/chat/sessions/${sessionId}/stream?message=${encodeURIComponent(message)}`;

        // Create EventSource for SSE (connects to our Next.js proxy)
        const eventSource = new EventSource(url);
        eventSourceRef.current = eventSource;

        // Handle incoming messages
        eventSource.onmessage = function handleMessage(event) {
          try {
            const chunk = JSON.parse(event.data) as StreamChunk;

            // Reset retry count on successful message
            if (retryCountRef.current > 0) {
              retryCountRef.current = 0;
            }

            onChunk(chunk);

            // Close connection when stream ends
            if (chunk.type === "stream_end") {
              stopStreaming();
            }
          } catch (err) {
            console.error("Failed to parse SSE chunk:", err);
            const parseError =
              err instanceof Error
                ? err
                : new Error("Failed to parse stream chunk");
            setError(parseError);
          }
        };

        // Handle errors with retry logic
        eventSource.onerror = function handleError(event) {
          console.error("SSE error:", event);

          // Close current connection
          if (eventSourceRef.current) {
            eventSourceRef.current.close();
            eventSourceRef.current = null;
          }

          // Check if we should retry
          if (retryCountRef.current < MAX_RETRIES) {
            retryCountRef.current += 1;
            const retryDelay =
              INITIAL_RETRY_DELAY * Math.pow(2, retryCountRef.current - 1);

            toast.info("Connection interrupted", {
              description: `Retrying in ${retryDelay / 1000} seconds...`,
            });

            retryTimeoutRef.current = setTimeout(() => {
              // Retry by recursively calling sendMessage
              sendMessage(sessionId, message, onChunk).catch((err) => {
                console.error("Retry failed:", err);
              });
            }, retryDelay);
          } else {
            // Max retries exceeded
            const streamError = new Error(
              "Stream connection failed after multiple retries",
            );
            setError(streamError);
            toast.error("Connection Failed", {
              description:
                "Unable to connect to chat service. Please try again.",
            });
            stopStreaming();
          }
        };

        // Return a promise that resolves when streaming completes
        return new Promise<void>((resolve, reject) => {
          const cleanup = () => {
            eventSource.removeEventListener("message", messageHandler);
            eventSource.removeEventListener("error", errorHandler);
          };

          const messageHandler = (event: MessageEvent) => {
            try {
              const chunk = JSON.parse(event.data) as StreamChunk;
              if (chunk.type === "stream_end") {
                cleanup();
                resolve();
              } else if (chunk.type === "error") {
                cleanup();
                reject(
                  new Error(chunk.message || chunk.content || "Stream error"),
                );
              }
            } catch (_err) {
              // Already handled in onmessage
            }
          };

          const errorHandler = () => {
            cleanup();
            reject(new Error("Stream connection error"));
          };

          eventSource.addEventListener("message", messageHandler);
          eventSource.addEventListener("error", errorHandler);
        });
      } catch (err) {
        const streamError =
          err instanceof Error ? err : new Error("Failed to start stream");
        setError(streamError);
        setIsStreaming(false);
        throw streamError;
      }
    },
    [stopStreaming],
  );

  return {
    isStreaming,
    error,
    sendMessage,
    stopStreaming,
  };
}
