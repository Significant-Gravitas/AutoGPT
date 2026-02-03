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
  /** Tracks if text has been finalized (to prevent duplicate messages from handleTextEnded and handleStreamEnd) */
  textFinalizedRef: MutableRefObject<boolean>;
  /** Tracks if stream has ended (to handle duplicate StreamFinish events during reconnection) */
  streamEndedRef: MutableRefObject<boolean>;
  setMessages: Dispatch<SetStateAction<ChatMessageData[]>>;
  setIsStreamingInitiated: Dispatch<SetStateAction<boolean>>;
  setIsRegionBlockedModalOpen: Dispatch<SetStateAction<boolean>>;
  sessionId: string;
  onOperationStarted?: () => void;
  onActiveTaskStarted?: (taskInfo: {
    taskId: string;
    operationId: string;
    toolName: string;
    toolCallId: string;
  }) => void;
}

export function isRegionBlockedError(chunk: StreamChunk): boolean {
  if (chunk.code === "MODEL_NOT_AVAILABLE_REGION") return true;
  const message = chunk.message || chunk.content;
  if (typeof message !== "string") return false;
  return message.toLowerCase().includes("not available in your region");
}

/**
 * Get a user-friendly error message based on the error code.
 * Returns undefined if the code should use the default message.
 */
export function getUserFriendlyErrorMessage(
  code: string | undefined,
): string | undefined {
  switch (code) {
    case "TASK_EXPIRED":
      return "This operation has expired. Please try again.";
    case "TASK_NOT_FOUND":
      return "Could not find the requested operation.";
    case "ACCESS_DENIED":
      return "You do not have access to this operation.";
    case "QUEUE_OVERFLOW":
      return "Connection was interrupted. Please refresh to continue.";
    case "MODEL_NOT_AVAILABLE_REGION":
      return "This model is not available in your region.";
    default:
      return undefined;
  }
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
  // Check if text was already finalized to prevent duplicate messages
  if (deps.textFinalizedRef.current) {
    return;
  }

  const completedText = deps.streamingChunksRef.current.join("");
  if (completedText.trim()) {
    // Mark text as finalized before adding message
    deps.textFinalizedRef.current = true;

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
  // Use deterministic fallback instead of Date.now() to ensure same ID on replay
  const toolId =
    chunk.tool_id ||
    `tool-${deps.sessionId}-${chunk.idx ?? "unknown"}-${chunk.tool_name || "unknown"}`;

  const toolCallMessage: Extract<ChatMessageData, { type: "tool_call" }> = {
    type: "tool_call",
    toolId,
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

// Types that represent a tool response and should be deduplicated by toolId
const TOOL_RESPONSE_TYPES = new Set([
  "tool_response",
  "operation_started",
  "operation_pending",
  "operation_in_progress",
  "execution_started",
  "agent_carousel",
  "clarification_needed",
]);

function hasResponseForTool(
  messages: ChatMessageData[],
  toolId: string,
): boolean {
  return messages.some((msg) => {
    if (!TOOL_RESPONSE_TYPES.has(msg.type)) return false;
    // Check various toolId field names used by different response types
    const msgToolId =
      (msg as { toolId?: string }).toolId ||
      (msg as { toolCallId?: string }).toolCallId;
    return msgToolId === toolId;
  });
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
        deps.setMessages((prev) => {
          // Check for duplicate inputs_needed message
          const exists = prev.some((msg) => msg.type === "inputs_needed");
          if (exists) return prev;
          return [...prev, inputsMessage];
        });
      }
      const credentialsMessage = extractCredentialsNeeded(
        parsedResult,
        chunk.tool_name,
      );
      if (credentialsMessage) {
        deps.setMessages((prev) => {
          // Check for duplicate credentials_needed message
          const exists = prev.some((msg) => msg.type === "credentials_needed");
          if (exists) return prev;
          return [...prev, credentialsMessage];
        });
      }
    }
    return;
  }
  // Trigger polling and store task info when operation_started is received
  if (responseMessage.type === "operation_started") {
    deps.onOperationStarted?.();
    // Store task info for SSE reconnection if taskId is present
    const taskId = (responseMessage as { taskId?: string }).taskId;
    if (taskId && deps.onActiveTaskStarted) {
      deps.onActiveTaskStarted({
        taskId,
        operationId:
          (responseMessage as { operationId?: string }).operationId || "",
        toolName: (responseMessage as { toolName?: string }).toolName || "",
        toolCallId: (responseMessage as { toolId?: string }).toolId || "",
      });
    }
  }

  deps.setMessages((prev) => {
    const toolCallIndex = prev.findIndex(
      (msg) => msg.type === "tool_call" && msg.toolId === chunk.tool_id,
    );
    // Check if any response type already exists for this tool
    if (hasResponseForTool(prev, chunk.tool_id!)) {
      return prev;
    }
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
  deps.setMessages((prev) => {
    // Check for duplicate login_needed message
    const exists = prev.some((msg) => msg.type === "login_needed");
    if (exists) return prev;
    return [...prev, loginNeededMessage];
  });
}

export function handleStreamEnd(
  _chunk: StreamChunk,
  deps: HandlerDependencies,
) {
  // Idempotent check - ignore duplicate finish events (can happen during reconnection)
  if (deps.streamEndedRef.current) {
    return;
  }
  deps.streamEndedRef.current = true;

  const completedContent = deps.streamingChunksRef.current.join("");
  if (!completedContent.trim() && !deps.hasResponseRef.current) {
    deps.setMessages((prev) => {
      // Check for duplicate "No response" message
      const exists = prev.some(
        (msg) =>
          msg.type === "message" &&
          msg.role === "assistant" &&
          msg.content === "No response received. Please try again.",
      );
      if (exists) return prev;
      return [
        ...prev,
        {
          type: "message",
          role: "assistant",
          content: "No response received. Please try again.",
          timestamp: new Date(),
        },
      ];
    });
  }
  // Only add message if text wasn't already finalized by handleTextEnded
  if (completedContent.trim() && !deps.textFinalizedRef.current) {
    // Mark as finalized BEFORE adding to prevent handleTextEnded from duplicating
    deps.textFinalizedRef.current = true;

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
  if (isRegionBlockedError(chunk)) {
    deps.setIsRegionBlockedModalOpen(true);
  }
  deps.setIsStreamingInitiated(false);
  deps.setHasTextChunks(false);
  deps.setStreamingChunks([]);
  deps.streamingChunksRef.current = [];
  // Reset flags since stream ended with error
  deps.textFinalizedRef.current = false;
  deps.streamEndedRef.current = true; // Mark as ended to prevent duplicate processing
}

/**
 * Get the display message for an error chunk, using user-friendly messages
 * for known error codes.
 */
export function getErrorDisplayMessage(chunk: StreamChunk): string {
  const friendlyMessage = getUserFriendlyErrorMessage(chunk.code);
  if (friendlyMessage) {
    return friendlyMessage;
  }
  return chunk.message || chunk.content || "An error occurred";
}
