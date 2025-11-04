import type { Dispatch, SetStateAction, MutableRefObject } from "react";
import type { StreamChunk } from "@/hooks/useChatStream";
import type { ChatMessageData } from "@/components/molecules/ChatMessage/useChatMessage";
import { parseToolResponse, extractCredentialsNeeded } from "./helpers";

/**
 * Handler dependencies - all state setters and refs needed by handlers.
 */
export interface HandlerDependencies {
  setHasTextChunks: Dispatch<SetStateAction<boolean>>;
  setStreamingChunks: Dispatch<SetStateAction<string[]>>;
  streamingChunksRef: MutableRefObject<string[]>;
  setMessages: Dispatch<SetStateAction<ChatMessageData[]>>;
  sessionId: string;
}

/**
 * Handles text_chunk events by accumulating streaming text.
 * Updates both the state and ref to prevent stale closures.
 */
export function handleTextChunk(
  chunk: StreamChunk,
  deps: HandlerDependencies,
): void {
  if (!chunk.content) return;

  deps.setHasTextChunks(true);
  deps.setStreamingChunks((prev) => {
    const updated = [...prev, chunk.content!];
    deps.streamingChunksRef.current = updated;
    return updated;
  });
}

/**
 * Handles text_ended events by saving completed text as assistant message.
 * Clears streaming state after saving the message.
 */
export function handleTextEnded(
  _chunk: StreamChunk,
  deps: HandlerDependencies,
): void {
  console.log("[Text Ended] Saving streamed text as assistant message");
  const completedText = deps.streamingChunksRef.current.join("");

  if (completedText.trim()) {
    const assistantMessage: ChatMessageData = {
      type: "message",
      role: "assistant",
      content: completedText,
      timestamp: new Date(),
    };
    deps.setMessages((prev) => [...prev, assistantMessage]);
  }

  // Clear streaming state
  deps.setStreamingChunks([]);
  deps.streamingChunksRef.current = [];
  deps.setHasTextChunks(false);
}

/**
 * Handles tool_call_start events by adding a ToolCallMessage to the UI.
 * Shows a loading state while the tool executes.
 */
export function handleToolCallStart(
  chunk: StreamChunk,
  deps: HandlerDependencies,
): void {
  const toolCallMessage: ChatMessageData = {
    type: "tool_call",
    toolId: chunk.tool_id || `tool-${Date.now()}-${chunk.idx || 0}`,
    toolName: chunk.tool_name || "Executing...",
    arguments: chunk.arguments || {},
    timestamp: new Date(),
  };
  deps.setMessages((prev) => [...prev, toolCallMessage]);

  console.log("[Tool Call Start]", {
    toolId: toolCallMessage.toolId,
    toolName: toolCallMessage.toolName,
    timestamp: new Date().toISOString(),
  });
}

/**
 * Handles tool_response events by replacing the matching tool_call message.
 * Parses the response and handles special cases like credential requirements.
 */
export function handleToolResponse(
  chunk: StreamChunk,
  deps: HandlerDependencies,
): void {
  console.log("[Tool Response] Received:", {
    toolId: chunk.tool_id,
    toolName: chunk.tool_name,
    timestamp: new Date().toISOString(),
  });

  // Find the matching tool_call to get the tool name if missing
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

  // Use helper function to parse tool response
  const responseMessage = parseToolResponse(
    chunk.result!,
    chunk.tool_id!,
    toolName,
    new Date(),
  );

  // If helper returns null (setup_requirements), handle credentials
  if (!responseMessage) {
    // Parse for credentials check
    let parsedResult: Record<string, unknown> | null = null;
    try {
      parsedResult =
        typeof chunk.result === "string"
          ? JSON.parse(chunk.result)
          : (chunk.result as Record<string, unknown>);
    } catch {
      parsedResult = null;
    }

    // Check if this is get_required_setup_info with missing credentials
    if (
      chunk.tool_name === "get_required_setup_info" &&
      chunk.success &&
      parsedResult
    ) {
      const credentialsMessage = extractCredentialsNeeded(parsedResult);
      if (credentialsMessage) {
        deps.setMessages((prev) => [...prev, credentialsMessage]);
      }
    }

    // Don't add message if setup_requirements
    return;
  }

  // Replace the tool_call message with matching tool_id
  deps.setMessages((prev) => {
    // Find the tool_call with the matching tool_id
    const toolCallIndex = prev.findIndex(
      (msg) => msg.type === "tool_call" && msg.toolId === chunk.tool_id,
    );

    if (toolCallIndex !== -1) {
      const newMessages = [...prev];
      newMessages[toolCallIndex] = responseMessage;

      console.log(
        "[Tool Response] Replaced tool_call with matching tool_id:",
        chunk.tool_id,
        "at index:",
        toolCallIndex,
      );
      return newMessages;
    }

    console.warn(
      "[Tool Response] No tool_call found with tool_id:",
      chunk.tool_id,
      "appending instead",
    );
    return [...prev, responseMessage];
  });
}

/**
 * Handles login_needed events by adding a login prompt message.
 */
export function handleLoginNeeded(
  chunk: StreamChunk,
  deps: HandlerDependencies,
): void {
  const loginNeededMessage: ChatMessageData = {
    type: "login_needed",
    message:
      chunk.message || "Please sign in to use chat and agent features",
    sessionId: chunk.session_id || deps.sessionId,
    agentInfo: chunk.agent_info,
    timestamp: new Date(),
  };
  deps.setMessages((prev) => [...prev, loginNeededMessage]);
}

/**
 * Handles stream_end events by finalizing the streaming session.
 * Converts any remaining streaming chunks into a completed assistant message.
 */
export function handleStreamEnd(
  _chunk: StreamChunk,
  deps: HandlerDependencies,
): void {
  // Convert streaming chunks into a completed assistant message
  // Use ref to get the latest chunks value (not stale closure value)
  const completedContent = deps.streamingChunksRef.current.join("");

  if (completedContent) {
    const assistantMessage: ChatMessageData = {
      type: "message",
      role: "assistant",
      content: completedContent,
      timestamp: new Date(),
    };

    // Add the completed assistant message to local state
    deps.setMessages((prev) => {
      const updated = [...prev, assistantMessage];

      // Log final state using current messages from state
      console.log("[Stream End] Final state:", {
        localMessages: updated.map((m) => ({
          type: m.type,
          ...(m.type === "message" && {
            role: m.role,
            contentLength: m.content.length,
          }),
          ...(m.type === "tool_call" && {
            toolId: m.toolId,
            toolName: m.toolName,
          }),
          ...(m.type === "tool_response" && {
            toolId: m.toolId,
            toolName: m.toolName,
            success: m.success,
          }),
        })),
        streamingChunks: deps.streamingChunksRef.current,
        timestamp: new Date().toISOString(),
      });

      return updated;
    });
  }

  // Clear streaming state immediately now that we have the message
  deps.setStreamingChunks([]);
  deps.streamingChunksRef.current = [];
  deps.setHasTextChunks(false);

  // Messages are now in local state and will be displayed
  console.log("[Stream End] Stream complete, messages in local state");
}

/**
 * Handles error events by logging and showing error toast.
 */
export function handleError(chunk: StreamChunk, _deps: HandlerDependencies): void {
  const errorMessage = chunk.message || chunk.content || "An error occurred";
  console.error("Stream error:", errorMessage);
  // Note: Toast import removed to avoid circular dependencies
  // Error toasts should be shown at the hook level
}
