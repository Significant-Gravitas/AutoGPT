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
  const retryCountRef = useRef<number>(0);
  const retryTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);

  const stopStreaming = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
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
      context?: { url: string; content: string },
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
        const url = `/api/chat/sessions/${sessionId}/stream`;
        const body = JSON.stringify({
          message,
          is_user_message: isUserMessage,
          context: context || null,
        });

        const response = await fetch(url, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Accept: "text/event-stream",
          },
          body,
          signal: abortController.signal,
        });

        if (!response.ok) {
          const errorText = await response.text();
          throw new Error(errorText || `HTTP ${response.status}`);
        }

        if (!response.body) {
          throw new Error("Response body is null");
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        return new Promise<void>((resolve, reject) => {
          const cleanup = () => {
            reader.cancel().catch(() => {
              // Ignore cancel errors
            });
          };

          const readStream = async () => {
            try {
              while (true) {
                const { done, value } = await reader.read();

                if (done) {
                  cleanup();
                  stopStreaming();
                  resolve();
                  return;
                }

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split("\n");
                buffer = lines.pop() || "";

                for (const line of lines) {
                  if (line.startsWith("data: ")) {
                    const data = line.slice(6);
                    if (data === "[DONE]") {
                      cleanup();
                      stopStreaming();
                      resolve();
                      return;
                    }

                    try {
                      const chunk = JSON.parse(data) as StreamChunk;

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
                        return;
                      } else if (chunk.type === "error") {
                        cleanup();
                        reject(
                          new Error(
                            chunk.message || chunk.content || "Stream error",
                          ),
                        );
                        return;
                      }
                    } catch (err) {
                      // Skip invalid JSON lines
                      console.warn("Failed to parse SSE chunk:", err, data);
                    }
                  }
                }
              }
            } catch (err) {
              if (err instanceof Error && err.name === "AbortError") {
                cleanup();
                return;
              }

              const streamError =
                err instanceof Error ? err : new Error("Failed to read stream");

              if (retryCountRef.current < MAX_RETRIES) {
                retryCountRef.current += 1;
                const retryDelay =
                  INITIAL_RETRY_DELAY * Math.pow(2, retryCountRef.current - 1);

                toast.info("Connection interrupted", {
                  description: `Retrying in ${retryDelay / 1000} seconds...`,
                });

                retryTimeoutRef.current = setTimeout(() => {
                  sendMessage(
                    sessionId,
                    message,
                    onChunk,
                    isUserMessage,
                    context,
                  ).catch((_err) => {
                    // Retry failed
                  });
                }, retryDelay);
              } else {
                setError(streamError);
                toast.error("Connection Failed", {
                  description:
                    "Unable to connect to chat service. Please try again.",
                });
                cleanup();
                stopStreaming();
                reject(streamError);
              }
            }
          };

          readStream();
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
