import type { Dispatch, MutableRefObject, SetStateAction } from "react";
import { StreamChunk } from "../../useChatStream";
import type { ChatMessageData } from "../ChatMessage/useChatMessage";
import {
  extractCredentialsNeeded,
  extractInputsNeeded,
  parseToolResponse,
} from "./helpers";

export interface HandlerDependencies {
  setHasTextChunks: Dispatch<SetStateAction<boolean>>;
  setStreamingChunks: Dispatch<SetStateAction<string[]>>;
  streamingChunksRef: MutableRefObject<string[]>;
  setMessages: Dispatch<SetStateAction<ChatMessageData[]>>;
  setIsStreamingInitiated: Dispatch<SetStateAction<boolean>>;
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
  const completedText = deps.streamingChunksRef.current.join("");
  if (completedText.trim()) {
    deps.setMessages((prev) => {
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
  const toolCallMessage: ChatMessageData = {
    type: "tool_call",
    toolId: chunk.tool_id || `tool-${Date.now()}-${chunk.idx || 0}`,
    toolName: chunk.tool_name || "Executing...",
    arguments: chunk.arguments || {},
    timestamp: new Date(),
  };
  deps.setMessages((prev) => [...prev, toolCallMessage]);
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
  deps.setMessages((prev) => {
    const toolCallIndex = prev.findIndex(
      (msg) => msg.type === "tool_call" && msg.toolId === chunk.tool_id,
    );
    if (toolCallIndex !== -1) {
      const newMessages = [...prev];
      newMessages[toolCallIndex] = responseMessage;
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
  if (completedContent.trim()) {
    const assistantMessage: ChatMessageData = {
      type: "message",
      role: "assistant",
      content: completedContent,
      timestamp: new Date(),
    };
    deps.setMessages((prev) => [...prev, assistantMessage]);
  }
  deps.setStreamingChunks([]);
  deps.streamingChunksRef.current = [];
  deps.setHasTextChunks(false);
  deps.setIsStreamingInitiated(false);
}

export function handleError(chunk: StreamChunk, deps: HandlerDependencies) {
  const errorMessage = chunk.message || chunk.content || "An error occurred";
  console.error("Stream error:", errorMessage);
  deps.setIsStreamingInitiated(false);
  deps.setHasTextChunks(false);
  deps.setStreamingChunks([]);
  deps.streamingChunksRef.current = [];
}
