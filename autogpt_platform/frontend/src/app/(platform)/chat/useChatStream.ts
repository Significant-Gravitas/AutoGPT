import { useState, useCallback, useRef, useEffect } from "react";
import { toast } from "sonner";
import type { ToolArguments, ToolResult } from "@/types/chat";

const MAX_RETRIES = 3;
const INITIAL_RETRY_DELAY = 1000;

export interface StreamChunk {
  type:
    | "text_chunk"
    | "text_ended"
    | "tool_call"
    | "tool_call_start"
    | "tool_response"
    | "login_needed"
    | "need_login"
    | "credentials_needed"
    | "error"
    | "usage"
    | "stream_end";
  timestamp?: string;
  content?: string;
  message?: string;
  tool_id?: string;
  tool_name?: string;
  arguments?: ToolArguments;
  result?: ToolResult;
  success?: boolean;
  idx?: number;
  session_id?: string;
  agent_info?: {
    graph_id: string;
    name: string;
    trigger_type: string;
  };
  provider?: string;
  provider_name?: string;
  credential_type?: string;
  scopes?: string[];
  title?: string;
  [key: string]: unknown;
}

export function useChatStream() {
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const eventSourceRef = useRef<EventSource | null>(null);
  const retryCountRef = useRef<number>(0);
  const retryTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);

  const stopStreaming = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
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

  useEffect(() => {
    return () => {
      stopStreaming();
    };
  }, [stopStreaming]);

  const sendMessage = useCallback(
    async (
      sessionId: string,
      message: string,
      onChunk: (chunk: StreamChunk) => void,
      isUserMessage: boolean = true,
    ) => {
      stopStreaming();

      const abortController = new AbortController();
      abortControllerRef.current = abortController;

      if (abortController.signal.aborted) {
        return Promise.reject(new Error("Request aborted"));
      }

      retryCountRef.current = 0;
      setIsStreaming(true);
      setError(null);

      try {
        const url = `/api/chat/sessions/${sessionId}/stream?message=${encodeURIComponent(
          message,
        )}&is_user_message=${isUserMessage}`;

        const eventSource = new EventSource(url);
        eventSourceRef.current = eventSource;

        abortController.signal.addEventListener("abort", () => {
          eventSource.close();
          eventSourceRef.current = null;
        });

        return new Promise<void>((resolve, reject) => {
          const cleanup = () => {
            eventSource.removeEventListener("message", messageHandler);
            eventSource.removeEventListener("error", errorHandler);
          };

          const messageHandler = (event: MessageEvent) => {
            try {
              const chunk = JSON.parse(event.data) as StreamChunk;

              if (retryCountRef.current > 0) {
                retryCountRef.current = 0;
              }

              // Call the chunk handler
              onChunk(chunk);

              // Handle stream lifecycle
              if (chunk.type === "stream_end") {
                cleanup();
                stopStreaming();
                resolve();
              } else if (chunk.type === "error") {
                cleanup();
                reject(
                  new Error(chunk.message || chunk.content || "Stream error"),
                );
              }
            } catch (err) {
              const parseError =
                err instanceof Error
                  ? err
                  : new Error("Failed to parse stream chunk");
              setError(parseError);
              cleanup();
              reject(parseError);
            }
          };

          const errorHandler = () => {
            if (eventSourceRef.current) {
              eventSourceRef.current.close();
              eventSourceRef.current = null;
            }

            if (retryCountRef.current < MAX_RETRIES) {
              retryCountRef.current += 1;
              const retryDelay =
                INITIAL_RETRY_DELAY * Math.pow(2, retryCountRef.current - 1);

              toast.info("Connection interrupted", {
                description: `Retrying in ${retryDelay / 1000} seconds...`,
              });

              retryTimeoutRef.current = setTimeout(() => {
                sendMessage(sessionId, message, onChunk, isUserMessage).catch(
                  (_err) => {
                    // Retry failed
                  },
                );
              }, retryDelay);
            } else {
              const streamError = new Error(
                "Stream connection failed after multiple retries",
              );
              setError(streamError);
              toast.error("Connection Failed", {
                description:
                  "Unable to connect to chat service. Please try again.",
              });
              cleanup();
              stopStreaming();
              reject(streamError);
            }
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
