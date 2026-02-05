import type { ToolArguments, ToolResult } from "@/types/chat";

export type StreamStatus = "idle" | "streaming" | "completed" | "error";

export interface StreamChunk {
  type:
    | "stream_start"
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
  taskId?: string;
  timestamp?: string;
  content?: string;
  message?: string;
  code?: string;
  details?: Record<string, unknown>;
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

export type VercelStreamChunk =
  | { type: "start"; messageId: string; taskId?: string }
  | { type: "finish" }
  | { type: "text-start"; id: string }
  | { type: "text-delta"; id: string; delta: string }
  | { type: "text-end"; id: string }
  | { type: "tool-input-start"; toolCallId: string; toolName: string }
  | {
      type: "tool-input-available";
      toolCallId: string;
      toolName: string;
      input: Record<string, unknown>;
    }
  | {
      type: "tool-output-available";
      toolCallId: string;
      toolName?: string;
      output: unknown;
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

export interface ActiveStream {
  sessionId: string;
  abortController: AbortController;
  status: StreamStatus;
  startedAt: number;
  chunks: StreamChunk[];
  error?: Error;
  onChunkCallbacks: Set<(chunk: StreamChunk) => void>;
}

export interface StreamResult {
  sessionId: string;
  status: StreamStatus;
  chunks: StreamChunk[];
  completedAt: number;
  error?: Error;
}

export type StreamCompleteCallback = (sessionId: string) => void;

// Type guards for message types

/**
 * Check if a message has a toolId property.
 */
export function hasToolId<T extends { type: string }>(
  msg: T,
): msg is T & { toolId: string } {
  return (
    "toolId" in msg &&
    typeof (msg as Record<string, unknown>).toolId === "string"
  );
}

/**
 * Check if a message has an operationId property.
 */
export function hasOperationId<T extends { type: string }>(
  msg: T,
): msg is T & { operationId: string } {
  return (
    "operationId" in msg &&
    typeof (msg as Record<string, unknown>).operationId === "string"
  );
}

/**
 * Check if a message has a toolCallId property.
 */
export function hasToolCallId<T extends { type: string }>(
  msg: T,
): msg is T & { toolCallId: string } {
  return (
    "toolCallId" in msg &&
    typeof (msg as Record<string, unknown>).toolCallId === "string"
  );
}

/**
 * Check if a message is an operation message type.
 */
export function isOperationMessage<T extends { type: string }>(
  msg: T,
): msg is T & {
  type: "operation_started" | "operation_pending" | "operation_in_progress";
} {
  return (
    msg.type === "operation_started" ||
    msg.type === "operation_pending" ||
    msg.type === "operation_in_progress"
  );
}

/**
 * Get the tool ID from a message if available.
 * Checks toolId, operationId, and toolCallId properties.
 */
export function getToolIdFromMessage<T extends { type: string }>(
  msg: T,
): string | undefined {
  const record = msg as Record<string, unknown>;
  if (typeof record.toolId === "string") return record.toolId;
  if (typeof record.operationId === "string") return record.operationId;
  if (typeof record.toolCallId === "string") return record.toolCallId;
  return undefined;
}
