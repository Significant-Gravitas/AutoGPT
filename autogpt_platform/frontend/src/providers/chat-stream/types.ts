import type { StreamChunk } from "./stream-utils";

export type StreamStatus = "idle" | "streaming" | "completed" | "error";

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
