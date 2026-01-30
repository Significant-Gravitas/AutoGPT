import type { Dispatch, MutableRefObject, SetStateAction } from "react";
import { StreamChunk } from "../../useChatStream";
import type { ChatMessageData } from "../ChatMessage/useChatMessage";
import {
  extractCredentialsNeeded,
  extractInputsNeeded,
  parseToolResponse,
} from "./helpers";

function isToolCallMessage(
  message: ChatMessageData,
): message is Extract<ChatMessageData, { type: "tool_call" }> {
  return message.type === "tool_call";
}

export interface HandlerDependencies {
  setHasTextChunks: Dispatch<SetStateAction<boolean>>;
  setStreamingChunks: Dispatch<SetStateAction<string[]>>;
  streamingChunksRef: MutableRefObject<string[]>;
  hasResponseRef: MutableRefObject<boolean>;
  setMessages: Dispatch<SetStateAction<ChatMessageData[]>>;
  setIsStreamingInitiated: Dispatch<SetStateAction<boolean>>;
  setIsRegionBlockedModalOpen: Dispatch<SetStateAction<boolean>>;
  sessionId: string;
  onOperationStarted?: () => void;
}

export function isRegionBlockedError(chunk: StreamChunk): boolean {
  if (chunk.code === "MODEL_NOT_AVAILABLE_REGION") return true;
  const message = chunk.message || chunk.content;
  if (typeof message !== "string") return false;
  return message.toLowerCase().includes("not available in your region");
}

export function handleTextChunk(chunk: StreamChunk, deps: HandlerDependencies) {
  if (!chunk.content) return;
  deps.setHasTextChunks(true);
  deps.setStreamingChunks((prev) => {
    const updated = [...prev, chunk.content!];
    deps.streamingChunksRef.current = updated;
    return updated;
  });
}

export function handleTextEnded(
  _chunk: StreamChunk,
  deps: HandlerDependencies,
) {
  const completedText = deps.streamingChunksRef.current.join("");
  if (completedText.trim()) {
    deps.setMessages((prev) => {
      // Check if this exact message already exists to prevent duplicates
      const exists = prev.some(
        (msg) =>
          msg.type === "message" &&
          msg.role === "assistant" &&
          msg.content === completedText,
      );
      if (exists) return prev;

      const assistantMessage: ChatMessageData = {
        type: "message",
        role: "assistant",
        content: completedText,
        timestamp: new Date(),
      };
      return [...prev, assistantMessage];
    });
  }
  deps.setStreamingChunks([]);
  deps.streamingChunksRef.current = [];
  deps.setHasTextChunks(false);
}

export function handleToolCallStart(
  chunk: StreamChunk,
  deps: HandlerDependencies,
) {
  const toolCallMessage: Extract<ChatMessageData, { type: "tool_call" }> = {
    type: "tool_call",
    toolId: chunk.tool_id || `tool-${Date.now()}-${chunk.idx || 0}`,
    toolName: chunk.tool_name || "Executing",
    arguments: chunk.arguments || {},
    timestamp: new Date(),
  };

  function updateToolCallMessages(prev: ChatMessageData[]) {
    const existingIndex = prev.findIndex(function findToolCallIndex(msg) {
      return isToolCallMessage(msg) && msg.toolId === toolCallMessage.toolId;
    });
    if (existingIndex === -1) {
      return [...prev, toolCallMessage];
    }
    const nextMessages = [...prev];
    const existing = nextMessages[existingIndex];
    if (!isToolCallMessage(existing)) return prev;
    const nextArguments =
      toolCallMessage.arguments &&
      Object.keys(toolCallMessage.arguments).length > 0
        ? toolCallMessage.arguments
        : existing.arguments;
    nextMessages[existingIndex] = {
      ...existing,
      toolName: toolCallMessage.toolName || existing.toolName,
      arguments: nextArguments,
      timestamp: toolCallMessage.timestamp,
    };
    return nextMessages;
  }

  deps.setMessages(updateToolCallMessages);
}

export function handleToolResponse(
  chunk: StreamChunk,
  deps: HandlerDependencies,
) {
  let toolName = chunk.tool_name || "unknown";
  if (!chunk.tool_name || chunk.tool_name === "unknown") {
    deps.setMessages((prev) => {
      const matchingToolCall = [...prev]
        .reverse()
        .find(
          (msg) => msg.type === "tool_call" && msg.toolId === chunk.tool_id,
        );
      if (matchingToolCall && matchingToolCall.type === "tool_call") {
        toolName = matchingToolCall.toolName;
      }
      return prev;
    });
  }
  const responseMessage = parseToolResponse(
    chunk.result!,
    chunk.tool_id!,
    toolName,
    new Date(),
  );
  if (!responseMessage) {
    let parsedResult: Record<string, unknown> | null = null;
    try {
      parsedResult =
        typeof chunk.result === "string"
          ? JSON.parse(chunk.result)
          : (chunk.result as Record<string, unknown>);
    } catch {
      parsedResult = null;
    }
    if (
      (chunk.tool_name === "run_agent" || chunk.tool_name === "run_block") &&
      chunk.success &&
      parsedResult?.type === "setup_requirements"
    ) {
      const inputsMessage = extractInputsNeeded(parsedResult, chunk.tool_name);
      if (inputsMessage) {
        deps.setMessages((prev) => [...prev, inputsMessage]);
      }
      const credentialsMessage = extractCredentialsNeeded(
        parsedResult,
        chunk.tool_name,
      );
      if (credentialsMessage) {
        deps.setMessages((prev) => [...prev, credentialsMessage]);
      }
    }
    return;
  }
  // Trigger polling when operation_started is received
  if (responseMessage.type === "operation_started") {
    deps.onOperationStarted?.();
  }

  deps.setMessages((prev) => {
    const toolCallIndex = prev.findIndex(
      (msg) => msg.type === "tool_call" && msg.toolId === chunk.tool_id,
    );
    const hasResponse = prev.some(
      (msg) => msg.type === "tool_response" && msg.toolId === chunk.tool_id,
    );
    if (hasResponse) return prev;
    if (toolCallIndex !== -1) {
      const newMessages = [...prev];
      newMessages.splice(toolCallIndex + 1, 0, responseMessage);
      return newMessages;
    }
    return [...prev, responseMessage];
  });
}

export function handleLoginNeeded(
  chunk: StreamChunk,
  deps: HandlerDependencies,
) {
  const loginNeededMessage: ChatMessageData = {
    type: "login_needed",
    toolName: "login_needed",
    message: chunk.message || "Please sign in to use chat and agent features",
    sessionId: chunk.session_id || deps.sessionId,
    agentInfo: chunk.agent_info,
    timestamp: new Date(),
  };
  deps.setMessages((prev) => [...prev, loginNeededMessage]);
}

export function handleStreamEnd(
  _chunk: StreamChunk,
  deps: HandlerDependencies,
) {
  const completedContent = deps.streamingChunksRef.current.join("");
  if (!completedContent.trim() && !deps.hasResponseRef.current) {
    deps.setMessages((prev) => [
      ...prev,
      {
        type: "message",
        role: "assistant",
        content: "No response received. Please try again.",
        timestamp: new Date(),
      },
    ]);
  }
  if (completedContent.trim()) {
    deps.setMessages((prev) => {
      // Check if this exact message already exists to prevent duplicates
      const exists = prev.some(
        (msg) =>
          msg.type === "message" &&
          msg.role === "assistant" &&
          msg.content === completedContent,
      );
      if (exists) return prev;

      const assistantMessage: ChatMessageData = {
        type: "message",
        role: "assistant",
        content: completedContent,
        timestamp: new Date(),
      };
      return [...prev, assistantMessage];
    });
  }
  deps.setStreamingChunks([]);
  deps.streamingChunksRef.current = [];
  deps.setHasTextChunks(false);
  deps.setIsStreamingInitiated(false);
}

export function handleError(chunk: StreamChunk, deps: HandlerDependencies) {
  const errorMessage = chunk.message || chunk.content || "An error occurred";
  console.error("Stream error:", errorMessage);
  if (isRegionBlockedError(chunk)) {
    deps.setIsRegionBlockedModalOpen(true);
  }
  deps.setIsStreamingInitiated(false);
  deps.setHasTextChunks(false);
  deps.setStreamingChunks([]);
  deps.streamingChunksRef.current = [];
}
