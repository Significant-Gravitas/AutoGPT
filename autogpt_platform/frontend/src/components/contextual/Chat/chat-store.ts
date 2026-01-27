"use client";

import { create } from "zustand";
import type {
  ActiveStream,
  StreamChunk,
  StreamCompleteCallback,
  StreamResult,
  StreamStatus,
} from "./chat-types";
import { executeStream } from "./stream-executor";

const COMPLETED_STREAM_TTL = 5 * 60 * 1000; // 5 minutes

interface ChatStoreState {
  activeStreams: Map<string, ActiveStream>;
  completedStreams: Map<string, StreamResult>;
  activeSessions: Set<string>;
  streamCompleteCallbacks: Set<StreamCompleteCallback>;
}

interface ChatStoreActions {
  startStream: (
    sessionId: string,
    message: string,
    isUserMessage: boolean,
    context?: { url: string; content: string },
    onChunk?: (chunk: StreamChunk) => void,
  ) => Promise<void>;
  stopStream: (sessionId: string) => void;
  subscribeToStream: (
    sessionId: string,
    onChunk: (chunk: StreamChunk) => void,
  ) => () => void;
  getStreamStatus: (sessionId: string) => StreamStatus;
  getCompletedStream: (sessionId: string) => StreamResult | undefined;
  clearCompletedStream: (sessionId: string) => void;
  isStreaming: (sessionId: string) => boolean;
  registerActiveSession: (sessionId: string) => void;
  unregisterActiveSession: (sessionId: string) => void;
  isSessionActive: (sessionId: string) => boolean;
  onStreamComplete: (callback: StreamCompleteCallback) => () => void;
}

type ChatStore = ChatStoreState & ChatStoreActions;

function notifyStreamComplete(
  callbacks: Set<StreamCompleteCallback>,
  sessionId: string,
) {
  for (const callback of callbacks) {
    try {
      callback(sessionId);
    } catch (err) {
      console.warn("[ChatStore] Stream complete callback error:", err);
    }
  }
}

function cleanupCompletedStreams(completedStreams: Map<string, StreamResult>) {
  const now = Date.now();
  for (const [sessionId, result] of completedStreams) {
    if (now - result.completedAt > COMPLETED_STREAM_TTL) {
      completedStreams.delete(sessionId);
    }
  }
}

function moveToCompleted(
  activeStreams: Map<string, ActiveStream>,
  completedStreams: Map<string, StreamResult>,
  streamCompleteCallbacks: Set<StreamCompleteCallback>,
  sessionId: string,
) {
  const stream = activeStreams.get(sessionId);
  if (!stream) return;

  const result: StreamResult = {
    sessionId,
    status: stream.status,
    chunks: stream.chunks,
    completedAt: Date.now(),
    error: stream.error,
  };

  completedStreams.set(sessionId, result);
  activeStreams.delete(sessionId);
  cleanupCompletedStreams(completedStreams);

  if (stream.status === "completed") {
    notifyStreamComplete(streamCompleteCallbacks, sessionId);
  }
}

export const useChatStore = create<ChatStore>((set, get) => ({
  activeStreams: new Map(),
  completedStreams: new Map(),
  activeSessions: new Set(),
  streamCompleteCallbacks: new Set(),

  startStream: async function startStream(
    sessionId,
    message,
    isUserMessage,
    context,
    onChunk,
  ) {
    const { activeStreams, completedStreams, streamCompleteCallbacks } = get();

    const existingStream = activeStreams.get(sessionId);
    if (existingStream) {
      existingStream.abortController.abort();
      moveToCompleted(
        activeStreams,
        completedStreams,
        streamCompleteCallbacks,
        sessionId,
      );
    }

    const abortController = new AbortController();
    const initialCallbacks = new Set<(chunk: StreamChunk) => void>();
    if (onChunk) initialCallbacks.add(onChunk);

    const stream: ActiveStream = {
      sessionId,
      abortController,
      status: "streaming",
      startedAt: Date.now(),
      chunks: [],
      onChunkCallbacks: initialCallbacks,
    };

    activeStreams.set(sessionId, stream);

    try {
      await executeStream(stream, message, isUserMessage, context);
    } finally {
      if (onChunk) stream.onChunkCallbacks.delete(onChunk);
      if (stream.status !== "streaming") {
        moveToCompleted(
          activeStreams,
          completedStreams,
          streamCompleteCallbacks,
          sessionId,
        );
      }
    }
  },

  stopStream: function stopStream(sessionId) {
    const { activeStreams, completedStreams, streamCompleteCallbacks } = get();
    const stream = activeStreams.get(sessionId);
    if (stream) {
      stream.abortController.abort();
      stream.status = "completed";
      moveToCompleted(
        activeStreams,
        completedStreams,
        streamCompleteCallbacks,
        sessionId,
      );
    }
  },

  subscribeToStream: function subscribeToStream(sessionId, onChunk) {
    const { activeStreams, completedStreams } = get();

    const stream = activeStreams.get(sessionId);
    if (stream) {
      for (const chunk of stream.chunks) {
        onChunk(chunk);
      }
      stream.onChunkCallbacks.add(onChunk);
      return function unsubscribe() {
        stream.onChunkCallbacks.delete(onChunk);
      };
    }

    const completed = completedStreams.get(sessionId);
    if (completed) {
      for (const chunk of completed.chunks) {
        onChunk(chunk);
      }
    }

    return function noop() {};
  },

  getStreamStatus: function getStreamStatus(sessionId) {
    const { activeStreams, completedStreams } = get();

    const active = activeStreams.get(sessionId);
    if (active) return active.status;

    const completed = completedStreams.get(sessionId);
    if (completed) return completed.status;

    return "idle";
  },

  getCompletedStream: function getCompletedStream(sessionId) {
    return get().completedStreams.get(sessionId);
  },

  clearCompletedStream: function clearCompletedStream(sessionId) {
    get().completedStreams.delete(sessionId);
  },

  isStreaming: function isStreaming(sessionId) {
    const stream = get().activeStreams.get(sessionId);
    return stream?.status === "streaming";
  },

  registerActiveSession: function registerActiveSession(sessionId) {
    get().activeSessions.add(sessionId);
  },

  unregisterActiveSession: function unregisterActiveSession(sessionId) {
    get().activeSessions.delete(sessionId);
  },

  isSessionActive: function isSessionActive(sessionId) {
    return get().activeSessions.has(sessionId);
  },

  onStreamComplete: function onStreamComplete(callback) {
    const { streamCompleteCallbacks } = get();
    streamCompleteCallbacks.add(callback);
    return function unsubscribe() {
      streamCompleteCallbacks.delete(callback);
    };
  },
}));
