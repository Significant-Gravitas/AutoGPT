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
    skipReplay?: boolean,
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

function cleanupExpiredStreams(
  completedStreams: Map<string, StreamResult>,
): Map<string, StreamResult> {
  const now = Date.now();
  const cleaned = new Map(completedStreams);
  for (const [sessionId, result] of cleaned) {
    if (now - result.completedAt > COMPLETED_STREAM_TTL) {
      cleaned.delete(sessionId);
    }
  }
  return cleaned;
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
    const state = get();
    const newActiveStreams = new Map(state.activeStreams);
    let newCompletedStreams = new Map(state.completedStreams);
    const callbacks = state.streamCompleteCallbacks;

    const existingStream = newActiveStreams.get(sessionId);
    if (existingStream) {
      existingStream.abortController.abort();
      const normalizedStatus =
        existingStream.status === "streaming"
          ? "completed"
          : existingStream.status;
      const result: StreamResult = {
        sessionId,
        status: normalizedStatus,
        chunks: existingStream.chunks,
        completedAt: Date.now(),
        error: existingStream.error,
      };
      newCompletedStreams.set(sessionId, result);
      newActiveStreams.delete(sessionId);
      newCompletedStreams = cleanupExpiredStreams(newCompletedStreams);
      if (normalizedStatus === "completed" || normalizedStatus === "error") {
        notifyStreamComplete(callbacks, sessionId);
      }
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

    newActiveStreams.set(sessionId, stream);
    set({
      activeStreams: newActiveStreams,
      completedStreams: newCompletedStreams,
    });

    try {
      await executeStream(stream, message, isUserMessage, context);
    } finally {
      if (onChunk) stream.onChunkCallbacks.delete(onChunk);
      if (stream.status !== "streaming") {
        const currentState = get();
        const finalActiveStreams = new Map(currentState.activeStreams);
        let finalCompletedStreams = new Map(currentState.completedStreams);

        const storedStream = finalActiveStreams.get(sessionId);
        if (storedStream === stream) {
          const result: StreamResult = {
            sessionId,
            status: stream.status,
            chunks: stream.chunks,
            completedAt: Date.now(),
            error: stream.error,
          };
          finalCompletedStreams.set(sessionId, result);
          finalActiveStreams.delete(sessionId);
          finalCompletedStreams = cleanupExpiredStreams(finalCompletedStreams);
          set({
            activeStreams: finalActiveStreams,
            completedStreams: finalCompletedStreams,
          });
          if (stream.status === "completed" || stream.status === "error") {
            notifyStreamComplete(
              currentState.streamCompleteCallbacks,
              sessionId,
            );
          }
        }
      }
    }
  },

  stopStream: function stopStream(sessionId) {
    const state = get();
    const stream = state.activeStreams.get(sessionId);
    if (!stream) return;

    stream.abortController.abort();
    stream.status = "completed";

    const newActiveStreams = new Map(state.activeStreams);
    let newCompletedStreams = new Map(state.completedStreams);

    const result: StreamResult = {
      sessionId,
      status: stream.status,
      chunks: stream.chunks,
      completedAt: Date.now(),
      error: stream.error,
    };
    newCompletedStreams.set(sessionId, result);
    newActiveStreams.delete(sessionId);
    newCompletedStreams = cleanupExpiredStreams(newCompletedStreams);

    set({
      activeStreams: newActiveStreams,
      completedStreams: newCompletedStreams,
    });

    notifyStreamComplete(state.streamCompleteCallbacks, sessionId);
  },

  subscribeToStream: function subscribeToStream(
    sessionId,
    onChunk,
    skipReplay = false,
  ) {
    const state = get();
    const stream = state.activeStreams.get(sessionId);

    if (stream) {
      if (!skipReplay) {
        for (const chunk of stream.chunks) {
          onChunk(chunk);
        }
      }

      stream.onChunkCallbacks.add(onChunk);

      return function unsubscribe() {
        stream.onChunkCallbacks.delete(onChunk);
      };
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
    const state = get();
    if (!state.completedStreams.has(sessionId)) return;

    const newCompletedStreams = new Map(state.completedStreams);
    newCompletedStreams.delete(sessionId);
    set({ completedStreams: newCompletedStreams });
  },

  isStreaming: function isStreaming(sessionId) {
    const stream = get().activeStreams.get(sessionId);
    return stream?.status === "streaming";
  },

  registerActiveSession: function registerActiveSession(sessionId) {
    const state = get();
    if (state.activeSessions.has(sessionId)) return;

    const newActiveSessions = new Set(state.activeSessions);
    newActiveSessions.add(sessionId);
    set({ activeSessions: newActiveSessions });
  },

  unregisterActiveSession: function unregisterActiveSession(sessionId) {
    const state = get();
    if (!state.activeSessions.has(sessionId)) return;

    const newActiveSessions = new Set(state.activeSessions);
    newActiveSessions.delete(sessionId);
    set({ activeSessions: newActiveSessions });
  },

  isSessionActive: function isSessionActive(sessionId) {
    return get().activeSessions.has(sessionId);
  },

  onStreamComplete: function onStreamComplete(callback) {
    const state = get();
    const newCallbacks = new Set(state.streamCompleteCallbacks);
    newCallbacks.add(callback);
    set({ streamCompleteCallbacks: newCallbacks });

    return function unsubscribe() {
      const currentState = get();
      const cleanedCallbacks = new Set(currentState.streamCompleteCallbacks);
      cleanedCallbacks.delete(callback);
      set({ streamCompleteCallbacks: cleanedCallbacks });
    };
  },
}));
