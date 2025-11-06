import type { Dispatch, SetStateAction, MutableRefObject } from "react";
import type { StreamChunk } from "@/app/(platform)/chat/useChatStream";
import type { ChatMessageData } from "@/app/(platform)/chat/components/ChatMessage/useChatMessage";
import { parseToolResponse, extractCredentialsNeeded } from "./helpers";

export interface HandlerDependencies {
  setHasTextChunks: Dispatch<SetStateAction<boolean>>;
  setStreamingChunks: Dispatch<SetStateAction<string[]>>;
  streamingChunksRef: MutableRefObject<string[]>;
  setMessages: Dispatch<SetStateAction<ChatMessageData[]>>;
  sessionId: string;
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
  deps.setStreamingChunks([]);
  deps.streamingChunksRef.current = [];
  deps.setHasTextChunks(false);
}

export function handleToolCallStart(
  chunk: StreamChunk,
  deps: HandlerDependencies,
) {
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

export function handleToolResponse(
  chunk: StreamChunk,
  deps: HandlerDependencies,
) {
  console.log("[Tool Response] Received:", {
    toolId: chunk.tool_id,
    toolName: chunk.tool_name,
    timestamp: new Date().toISOString(),
  });
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
      chunk.tool_name === "get_required_setup_info" &&
      chunk.success &&
      parsedResult
    ) {
      const credentialsMessage = extractCredentialsNeeded(parsedResult);
      if (credentialsMessage) {
        deps.setMessages((prev) => [...prev, credentialsMessage]);
      }
    }
    return;
  }
  deps.setMessages((prev) => {
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

export function handleLoginNeeded(
  chunk: StreamChunk,
  deps: HandlerDependencies,
) {
  const loginNeededMessage: ChatMessageData = {
    type: "login_needed",
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
  // Only save message if there are uncommitted chunks
  // (text_ended already saved if there were tool calls)
  if (completedContent.trim()) {
    console.log(
      "[Stream End] Saving remaining streamed text as assistant message",
    );
    const assistantMessage: ChatMessageData = {
      type: "message",
      role: "assistant",
      content: completedContent,
      timestamp: new Date(),
    };
    deps.setMessages((prev) => {
      const updated = [...prev, assistantMessage];
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
  } else {
    console.log("[Stream End] No uncommitted chunks, message already saved");
  }
  deps.setStreamingChunks([]);
  deps.streamingChunksRef.current = [];
  deps.setHasTextChunks(false);
  console.log("[Stream End] Stream complete, messages in local state");
}

export function handleError(chunk: StreamChunk, _deps: HandlerDependencies) {
  const errorMessage = chunk.message || chunk.content || "An error occurred";
  console.error("Stream error:", errorMessage);
}
