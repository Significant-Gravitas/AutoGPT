/**
 * useChatContainerAiSdk - ChatContainer hook using Vercel AI SDK
 *
 * This is a drop-in replacement for useChatContainer that uses @ai-sdk/react
 * instead of the custom streaming implementation. The API surface is identical
 * to enable easy A/B testing and gradual migration.
 */

import type { SessionDetailResponse } from "@/app/api/__generated__/models/sessionDetailResponse";
import { useEffect, useMemo, useRef } from "react";
import type { UIMessage } from "ai";
import { useAiSdkChat } from "../../useAiSdkChat";
import { usePageContext } from "../../usePageContext";
import type { ChatMessageData } from "../ChatMessage/useChatMessage";
import {
  filterAuthMessages,
  hasSentInitialPrompt,
  markInitialPromptSent,
  processInitialMessages,
} from "./helpers";

// Helper to convert backend messages to AI SDK UIMessage format
function convertToUIMessages(
  messages: SessionDetailResponse["messages"],
): UIMessage[] {
  const result: UIMessage[] = [];

  for (const msg of messages) {
    if (!msg.role || !msg.content) continue;

    // Create parts based on message type
    const parts: UIMessage["parts"] = [];

    if (msg.role === "user" || msg.role === "assistant") {
      if (typeof msg.content === "string") {
        parts.push({ type: "text", text: msg.content });
      }
    }

    // Handle tool calls in assistant messages
    if (msg.role === "assistant" && msg.tool_calls) {
      for (const toolCall of msg.tool_calls as Array<{
        id: string;
        type: string;
        function: { name: string; arguments: string };
      }>) {
        if (toolCall.type === "function") {
          let args = {};
          try {
            args = JSON.parse(toolCall.function.arguments);
          } catch {
            // Keep empty args
          }
          parts.push({
            type: `tool-${toolCall.function.name}` as `tool-${string}`,
            toolCallId: toolCall.id,
            toolName: toolCall.function.name,
            state: "input-available",
            input: args,
          } as UIMessage["parts"][number]);
        }
      }
    }

    // Handle tool responses
    if (msg.role === "tool" && msg.tool_call_id) {
      // Find the matching tool call to get the name
      const toolName = "unknown";
      let output: unknown = msg.content;
      try {
        output =
          typeof msg.content === "string"
            ? JSON.parse(msg.content)
            : msg.content;
      } catch {
        // Keep as string
      }

      parts.push({
        type: `tool-${toolName}` as `tool-${string}`,
        toolCallId: msg.tool_call_id as string,
        toolName,
        state: "output-available",
        output,
      } as UIMessage["parts"][number]);
    }

    if (parts.length > 0) {
      result.push({
        id: msg.id || `msg-${Date.now()}-${Math.random()}`,
        role: msg.role === "tool" ? "assistant" : (msg.role as "user" | "assistant"),
        parts,
        createdAt: msg.created_at ? new Date(msg.created_at as string) : new Date(),
      });
    }
  }

  return result;
}

interface Args {
  sessionId: string | null;
  initialMessages: SessionDetailResponse["messages"];
  initialPrompt?: string;
  onOperationStarted?: () => void;
}

export function useChatContainerAiSdk({
  sessionId,
  initialMessages,
  initialPrompt,
  onOperationStarted,
}: Args) {
  const { capturePageContext } = usePageContext();
  const sendMessageRef = useRef<
    (
      content: string,
      isUserMessage?: boolean,
      context?: { url: string; content: string },
    ) => Promise<void>
  >();

  // Convert initial messages to AI SDK format
  const uiMessages = useMemo(
    () => convertToUIMessages(initialMessages),
    [initialMessages],
  );

  const {
    messages: aiSdkMessages,
    streamingChunks,
    isStreaming,
    error,
    isRegionBlockedModalOpen,
    setIsRegionBlockedModalOpen,
    sendMessage,
    stopStreaming,
  } = useAiSdkChat({
    sessionId,
    initialMessages: uiMessages,
    onOperationStarted,
  });

  // Keep ref updated for initial prompt handling
  sendMessageRef.current = sendMessage;

  // Merge AI SDK messages with processed initial messages
  // This ensures we show both historical messages and new streaming messages
  const allMessages = useMemo(() => {
    const processedInitial = processInitialMessages(initialMessages);

    // Build a set of message keys for deduplication
    const seenKeys = new Set<string>();
    const result: ChatMessageData[] = [];

    // Add processed initial messages first
    for (const msg of processedInitial) {
      const key = getMessageKey(msg);
      if (!seenKeys.has(key)) {
        seenKeys.add(key);
        result.push(msg);
      }
    }

    // Add AI SDK messages that aren't duplicates
    for (const msg of aiSdkMessages) {
      const key = getMessageKey(msg);
      if (!seenKeys.has(key)) {
        seenKeys.add(key);
        result.push(msg);
      }
    }

    return result;
  }, [initialMessages, aiSdkMessages]);

  // Handle initial prompt
  useEffect(
    function handleInitialPrompt() {
      if (!initialPrompt || !sessionId) return;
      if (initialMessages.length > 0) return;
      if (hasSentInitialPrompt(sessionId)) return;

      markInitialPromptSent(sessionId);
      const context = capturePageContext();
      sendMessageRef.current?.(initialPrompt, true, context);
    },
    [initialPrompt, sessionId, initialMessages.length, capturePageContext],
  );

  // Send message with page context
  async function sendMessageWithContext(
    content: string,
    isUserMessage: boolean = true,
  ) {
    const context = capturePageContext();
    await sendMessage(content, isUserMessage, context);
  }

  function handleRegionModalOpenChange(open: boolean) {
    setIsRegionBlockedModalOpen(open);
  }

  function handleRegionModalClose() {
    setIsRegionBlockedModalOpen(false);
  }

  return {
    messages: filterAuthMessages(allMessages),
    streamingChunks,
    isStreaming,
    error,
    isRegionBlockedModalOpen,
    setIsRegionBlockedModalOpen,
    sendMessageWithContext,
    handleRegionModalOpenChange,
    handleRegionModalClose,
    sendMessage,
    stopStreaming,
  };
}

// Helper to generate deduplication key for a message
function getMessageKey(msg: ChatMessageData): string {
  if (msg.type === "message") {
    return `msg:${msg.role}:${msg.content}`;
  } else if (msg.type === "tool_call") {
    return `toolcall:${msg.toolId}`;
  } else if (msg.type === "tool_response") {
    return `toolresponse:${(msg as { toolId?: string }).toolId}`;
  } else if (
    msg.type === "operation_started" ||
    msg.type === "operation_pending" ||
    msg.type === "operation_in_progress"
  ) {
    const typedMsg = msg as {
      toolId?: string;
      operationId?: string;
      toolCallId?: string;
      toolName?: string;
    };
    return `op:${typedMsg.toolId || typedMsg.operationId || typedMsg.toolCallId || ""}:${typedMsg.toolName || ""}`;
  } else {
    return `${msg.type}:${JSON.stringify(msg).slice(0, 100)}`;
  }
}
