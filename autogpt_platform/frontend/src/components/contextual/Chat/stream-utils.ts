import type { ToolArguments, ToolResult } from "@/types/chat";
import type { StreamChunk, VercelStreamChunk } from "./chat-types";

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

export function isLegacyStreamChunk(
  chunk: StreamChunk | VercelStreamChunk,
): chunk is StreamChunk {
  return LEGACY_STREAM_TYPES.has(chunk.type as StreamChunk["type"]);
}

export function normalizeStreamChunk(
  chunk: StreamChunk | VercelStreamChunk,
): StreamChunk | null {
  if (isLegacyStreamChunk(chunk)) return chunk;

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
        arguments: chunk.input as ToolArguments,
      };
    case "tool-output-available":
      return {
        type: "tool_response",
        tool_id: chunk.toolCallId,
        tool_name: chunk.toolName,
        result: chunk.output as ToolResult,
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
      return null;
    case "tool-input-start":
      return {
        type: "tool_call_start",
        tool_id: chunk.toolCallId,
        tool_name: chunk.toolName,
        arguments: {},
      };
  }
}

export const MAX_RETRIES = 3;
export const INITIAL_RETRY_DELAY = 1000;

export function parseSSELine(line: string): string | null {
  if (line.startsWith("data: ")) return line.slice(6);
  return null;
}
