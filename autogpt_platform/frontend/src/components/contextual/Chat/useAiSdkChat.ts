"use client";

/**
 * useAiSdkChat - Vercel AI SDK integration for CoPilot Chat
 *
 * This hook wraps @ai-sdk/react's useChat to provide:
 * - Streaming chat with the existing Python backend (already AI SDK protocol compatible)
 * - Integration with existing session management
 * - Custom tool response parsing for AutoGPT-specific types
 * - Page context injection
 *
 * The Python backend already implements the AI SDK Data Stream Protocol (v1),
 * so this hook can communicate directly without any backend changes.
 */

import { useChat as useAiSdkChatBase } from "@ai-sdk/react";
import { DefaultChatTransport, type UIMessage } from "ai";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { toast } from "sonner";
import type { ChatMessageData } from "./components/ChatMessage/useChatMessage";

// Tool response types from the backend
type OperationType =
  | "operation_started"
  | "operation_pending"
  | "operation_in_progress";

interface ToolOutputBase {
  type: string;
  [key: string]: unknown;
}

interface UseAiSdkChatOptions {
  sessionId: string | null;
  initialMessages?: UIMessage[];
  onOperationStarted?: () => void;
  onStreamingChange?: (isStreaming: boolean) => void;
}

/**
 * Parse tool output from AI SDK message parts into ChatMessageData format
 */
function parseToolOutput(
  toolCallId: string,
  toolName: string,
  output: unknown,
): ChatMessageData | null {
  if (!output) return null;

  let parsed: ToolOutputBase;
  try {
    parsed =
      typeof output === "string"
        ? JSON.parse(output)
        : (output as ToolOutputBase);
  } catch {
    return null;
  }

  const type = parsed.type;

  // Handle operation status types
  if (
    type === "operation_started" ||
    type === "operation_pending" ||
    type === "operation_in_progress"
  ) {
    return {
      type: type as OperationType,
      toolId: toolCallId,
      toolName: toolName,
      operationId: (parsed.operation_id as string) || undefined,
      message: (parsed.message as string) || undefined,
      timestamp: new Date(),
    } as ChatMessageData;
  }

  // Handle agent carousel
  if (type === "agent_carousel" && Array.isArray(parsed.agents)) {
    return {
      type: "agent_carousel",
      toolId: toolCallId,
      toolName: toolName,
      agents: parsed.agents,
      timestamp: new Date(),
    } as ChatMessageData;
  }

  // Handle execution started
  if (type === "execution_started") {
    return {
      type: "execution_started",
      toolId: toolCallId,
      toolName: toolName,
      graphId: parsed.graph_id as string,
      graphVersion: parsed.graph_version as number,
      graphExecId: parsed.graph_exec_id as string,
      nodeExecIds: parsed.node_exec_ids as string[],
      timestamp: new Date(),
    } as ChatMessageData;
  }

  // Handle error responses
  if (type === "error") {
    return {
      type: "tool_response",
      toolId: toolCallId,
      toolName: toolName,
      result: parsed,
      success: false,
      timestamp: new Date(),
    } as ChatMessageData;
  }

  // Handle clarification questions
  if (type === "clarification_questions" && Array.isArray(parsed.questions)) {
    return {
      type: "clarification_questions",
      toolId: toolCallId,
      toolName: toolName,
      questions: parsed.questions,
      timestamp: new Date(),
    } as ChatMessageData;
  }

  // Handle credentials needed
  if (type === "credentials_needed" || type === "setup_requirements") {
    const credentials = parsed.credentials as
      | Array<{
          provider: string;
          provider_name: string;
          credential_type: string;
          scopes?: string[];
        }>
      | undefined;
    if (credentials && credentials.length > 0) {
      return {
        type: "credentials_needed",
        toolId: toolCallId,
        toolName: toolName,
        credentials: credentials,
        timestamp: new Date(),
      } as ChatMessageData;
    }
  }

  // Default: generic tool response
  return {
    type: "tool_response",
    toolId: toolCallId,
    toolName: toolName,
    result: parsed,
    success: true,
    timestamp: new Date(),
  } as ChatMessageData;
}

/**
 * Convert AI SDK UIMessage parts to ChatMessageData array
 */
function convertMessageToChatData(message: UIMessage): ChatMessageData[] {
  const result: ChatMessageData[] = [];

  for (const part of message.parts) {
    switch (part.type) {
      case "text":
        if (part.text.trim()) {
          result.push({
            type: "message",
            role: message.role as "user" | "assistant",
            content: part.text,
            timestamp: new Date(message.createdAt || Date.now()),
          });
        }
        break;

      default:
        // Handle tool parts (tool-*)
        if (part.type.startsWith("tool-")) {
          const toolPart = part as {
            type: string;
            toolCallId: string;
            toolName: string;
            state: string;
            input?: Record<string, unknown>;
            output?: unknown;
          };

          // Show tool call in progress
          if (
            toolPart.state === "input-streaming" ||
            toolPart.state === "input-available"
          ) {
            result.push({
              type: "tool_call",
              toolId: toolPart.toolCallId,
              toolName: toolPart.toolName,
              arguments: toolPart.input || {},
              timestamp: new Date(),
            });
          }

          // Parse tool output when available
          if (
            toolPart.state === "output-available" &&
            toolPart.output !== undefined
          ) {
            const parsed = parseToolOutput(
              toolPart.toolCallId,
              toolPart.toolName,
              toolPart.output,
            );
            if (parsed) {
              result.push(parsed);
            }
          }

          // Handle tool errors
          if (toolPart.state === "output-error") {
            result.push({
              type: "tool_response",
              toolId: toolPart.toolCallId,
              toolName: toolPart.toolName,
              response: {
                type: "error",
                message: (toolPart as { errorText?: string }).errorText,
              },
              success: false,
              timestamp: new Date(),
            } as ChatMessageData);
          }
        }
        break;
    }
  }

  return result;
}

export function useAiSdkChat({
  sessionId,
  initialMessages = [],
  onOperationStarted,
  onStreamingChange,
}: UseAiSdkChatOptions) {
  const [isRegionBlockedModalOpen, setIsRegionBlockedModalOpen] =
    useState(false);
  const previousSessionIdRef = useRef<string | null>(null);
  const hasNotifiedOperationRef = useRef<Set<string>>(new Set());

  // Create transport with session-specific endpoint
  const transport = useMemo(() => {
    if (!sessionId) return undefined;
    return new DefaultChatTransport({
      api: `/api/chat/sessions/${sessionId}/stream`,
      headers: {
        "Content-Type": "application/json",
      },
    });
  }, [sessionId]);

  const {
    messages: aiMessages,
    status,
    error,
    stop,
    setMessages,
    sendMessage: aiSendMessage,
  } = useAiSdkChatBase({
    transport,
    initialMessages,
    onError: (err) => {
      console.error("[useAiSdkChat] Error:", err);

      // Check for region blocking
      if (
        err.message?.toLowerCase().includes("not available in your region") ||
        (err as { code?: string }).code === "MODEL_NOT_AVAILABLE_REGION"
      ) {
        setIsRegionBlockedModalOpen(true);
        return;
      }

      toast.error("Chat Error", {
        description: err.message || "An error occurred",
      });
    },
    onFinish: ({ message }) => {
      console.info("[useAiSdkChat] Message finished:", {
        id: message.id,
        partsCount: message.parts.length,
      });
    },
  });

  // Track streaming status
  const isStreaming = status === "streaming" || status === "submitted";

  // Notify parent of streaming changes
  useEffect(() => {
    onStreamingChange?.(isStreaming);
  }, [isStreaming, onStreamingChange]);

  // Handle session changes - reset state
  useEffect(() => {
    if (sessionId === previousSessionIdRef.current) return;

    if (previousSessionIdRef.current && status === "streaming") {
      stop();
    }

    previousSessionIdRef.current = sessionId;
    hasNotifiedOperationRef.current = new Set();

    if (sessionId) {
      setMessages(initialMessages);
    }
  }, [sessionId, status, stop, setMessages, initialMessages]);

  // Convert AI SDK messages to ChatMessageData format
  const messages = useMemo(() => {
    const result: ChatMessageData[] = [];

    for (const message of aiMessages) {
      const converted = convertMessageToChatData(message);
      result.push(...converted);

      // Check for operation_started and notify
      for (const msg of converted) {
        if (
          msg.type === "operation_started" &&
          !hasNotifiedOperationRef.current.has(
            (msg as { toolId?: string }).toolId || "",
          )
        ) {
          hasNotifiedOperationRef.current.add(
            (msg as { toolId?: string }).toolId || "",
          );
          onOperationStarted?.();
        }
      }
    }

    return result;
  }, [aiMessages, onOperationStarted]);

  // Get streaming text chunks from the last assistant message
  const streamingChunks = useMemo(() => {
    if (!isStreaming) return [];

    const lastMessage = aiMessages[aiMessages.length - 1];
    if (!lastMessage || lastMessage.role !== "assistant") return [];

    const chunks: string[] = [];
    for (const part of lastMessage.parts) {
      if (part.type === "text" && part.text) {
        chunks.push(part.text);
      }
    }

    return chunks;
  }, [aiMessages, isStreaming]);

  // Send message with optional context
  const sendMessage = useCallback(
    async (
      content: string,
      isUserMessage: boolean = true,
      context?: { url: string; content: string },
    ) => {
      if (!sessionId || !transport) {
        console.error("[useAiSdkChat] Cannot send message: no session");
        return;
      }

      setIsRegionBlockedModalOpen(false);

      try {
        await aiSendMessage(
          { text: content },
          {
            body: {
              is_user_message: isUserMessage,
              context: context || null,
            },
          },
        );
      } catch (err) {
        console.error("[useAiSdkChat] Failed to send message:", err);

        if (err instanceof Error && err.name === "AbortError") return;

        toast.error("Failed to send message", {
          description:
            err instanceof Error ? err.message : "Failed to send message",
        });
      }
    },
    [sessionId, transport, aiSendMessage],
  );

  // Stop streaming
  const stopStreaming = useCallback(() => {
    stop();
  }, [stop]);

  return {
    messages,
    streamingChunks,
    isStreaming,
    error,
    status,
    isRegionBlockedModalOpen,
    setIsRegionBlockedModalOpen,
    sendMessage,
    stopStreaming,
    // Expose raw AI SDK state for advanced use cases
    aiMessages,
    setAiMessages: setMessages,
  };
}
