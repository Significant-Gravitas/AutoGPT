import type { ToolArguments, ToolResult } from "@/types/chat";
import { useCallback, useEffect, useRef, useState } from "react";
import { toast } from "sonner";

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

type VercelStreamChunk =
  | { type: "start"; messageId: string }
  | { type: "finish" }
  | { type: "text-start"; id: string }
  | { type: "text-delta"; id: string; delta: string }
  | { type: "text-end"; id: string }
  | { type: "tool-input-start"; toolCallId: string; toolName: string }
  | {
      type: "tool-input-available";
      toolCallId: string;
      toolName: string;
      input: ToolArguments;
    }
  | {
      type: "tool-output-available";
      toolCallId: string;
      toolName?: string;
      output: ToolResult;
      success?: boolean;
    }
  | {
      type: "usage";
      promptTokens: number;
      completionTokens: number;
      totalTokens: number;
    }
  | {
      type: "error";
      errorText: string;
      code?: string;
      details?: Record<string, unknown>;
    };

const LEGACY_STREAM_TYPES = new Set<StreamChunk["type"]>([
  "text_chunk",
  "text_ended",
  "tool_call",
  "tool_call_start",
  "tool_response",
  "login_needed",
  "need_login",
  "credentials_needed",
  "error",
  "usage",
  "stream_end",
]);

function isLegacyStreamChunk(
  chunk: StreamChunk | VercelStreamChunk,
): chunk is StreamChunk {
  return LEGACY_STREAM_TYPES.has(chunk.type as StreamChunk["type"]);
}

function normalizeStreamChunk(
  chunk: StreamChunk | VercelStreamChunk,
): StreamChunk | null {
  if (isLegacyStreamChunk(chunk)) {
    return chunk;
  }
  switch (chunk.type) {
    case "text-delta":
      return { type: "text_chunk", content: chunk.delta };
    case "text-end":
      return { type: "text_ended" };
    case "tool-input-available":
      return {
        type: "tool_call_start",
        tool_id: chunk.toolCallId,
        tool_name: chunk.toolName,
        arguments: chunk.input,
      };
    case "tool-output-available":
      return {
        type: "tool_response",
        tool_id: chunk.toolCallId,
        tool_name: chunk.toolName,
        result: chunk.output,
        success: chunk.success ?? true,
      };
    case "usage":
      return {
        type: "usage",
        promptTokens: chunk.promptTokens,
        completionTokens: chunk.completionTokens,
        totalTokens: chunk.totalTokens,
      };
    case "error":
      return {
        type: "error",
        message: chunk.errorText,
        code: chunk.code,
        details: chunk.details,
      };
    case "finish":
      return { type: "stream_end" };
    case "start":
    case "text-start":
    case "tool-input-start":
      return null;
  }
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
      isRetry: boolean = false,
    ) => {
      stopStreaming();

      const abortController = new AbortController();
      abortControllerRef.current = abortController;

      if (abortController.signal.aborted) {
        return Promise.reject(new Error("Request aborted"));
      }

      if (!isRetry) {
        retryCountRef.current = 0;
      }
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
          let didDispatchStreamEnd = false;

          function dispatchStreamEnd() {
            if (didDispatchStreamEnd) return;
            didDispatchStreamEnd = true;
            onChunk({ type: "stream_end" });
          }

          const cleanup = () => {
            reader.cancel().catch(() => {
              // Ignore cancel errors
            });
          };

          async function readStream() {
            try {
              while (true) {
                const { done, value } = await reader.read();

                if (done) {
                  cleanup();
                  dispatchStreamEnd();
                  retryCountRef.current = 0;
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
                      dispatchStreamEnd();
                      retryCountRef.current = 0;
                      stopStreaming();
                      resolve();
                      return;
                    }

                    try {
                      const rawChunk = JSON.parse(data) as
                        | StreamChunk
                        | VercelStreamChunk;
                      const chunk = normalizeStreamChunk(rawChunk);
                      if (!chunk) {
                        continue;
                      }

                      // Call the chunk handler
                      onChunk(chunk);

                      // Handle stream lifecycle
                      if (chunk.type === "stream_end") {
                        didDispatchStreamEnd = true;
                        cleanup();
                        retryCountRef.current = 0;
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
                    true,
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
                dispatchStreamEnd();
                retryCountRef.current = 0;
                stopStreaming();
                reject(streamError);
              }
            }
          }

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
