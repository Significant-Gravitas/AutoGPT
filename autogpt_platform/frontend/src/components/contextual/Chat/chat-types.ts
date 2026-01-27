import type { ToolArguments, ToolResult } from "@/types/chat";

export type StreamStatus = "idle" | "streaming" | "completed" | "error";

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
