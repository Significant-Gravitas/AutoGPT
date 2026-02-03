"use client";

import { create } from "zustand";
import {
  ACTIVE_TASK_TTL_MS,
  COMPLETED_STREAM_TTL_MS,
  INITIAL_STREAM_ID,
  STORAGE_KEY_ACTIVE_TASKS,
} from "./chat-constants";
import type {
  ActiveStream,
  StreamChunk,
  StreamCompleteCallback,
  StreamResult,
  StreamStatus,
} from "./chat-types";
import { executeStream, executeTaskReconnect } from "./stream-executor";

export interface ActiveTaskInfo {
  taskId: string;
  sessionId: string;
  operationId: string;
  toolName: string;
  lastMessageId: string;
  startedAt: number;
}

/** Load active tasks from localStorage */
function loadPersistedTasks(): Map<string, ActiveTaskInfo> {
  if (typeof window === "undefined") return new Map();
  try {
    const stored = localStorage.getItem(STORAGE_KEY_ACTIVE_TASKS);
    if (!stored) return new Map();
    const parsed = JSON.parse(stored) as Record<string, ActiveTaskInfo>;
    const now = Date.now();
    const tasks = new Map<string, ActiveTaskInfo>();
    // Filter out expired tasks
    for (const [sessionId, task] of Object.entries(parsed)) {
      if (now - task.startedAt < ACTIVE_TASK_TTL_MS) {
        tasks.set(sessionId, task);
      }
    }
    return tasks;
  } catch {
    return new Map();
  }
}

/** Save active tasks to localStorage */
function persistTasks(tasks: Map<string, ActiveTaskInfo>): void {
  if (typeof window === "undefined") return;
  try {
    const obj: Record<string, ActiveTaskInfo> = {};
    for (const [sessionId, task] of tasks) {
      obj[sessionId] = task;
    }
    localStorage.setItem(STORAGE_KEY_ACTIVE_TASKS, JSON.stringify(obj));
  } catch {
    // Ignore storage errors
  }
}

interface ChatStoreState {
  activeStreams: Map<string, ActiveStream>;
  completedStreams: Map<string, StreamResult>;
  activeSessions: Set<string>;
  streamCompleteCallbacks: Set<StreamCompleteCallback>;
  /** Active tasks for SSE reconnection - keyed by sessionId */
  activeTasks: Map<string, ActiveTaskInfo>;
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
  /** Track active task for SSE reconnection */
  setActiveTask: (
    sessionId: string,
    taskInfo: Omit<ActiveTaskInfo, "sessionId" | "startedAt">,
  ) => void;
  /** Get active task for a session */
  getActiveTask: (sessionId: string) => ActiveTaskInfo | undefined;
  /** Clear active task when operation completes */
  clearActiveTask: (sessionId: string) => void;
  /** Reconnect to an existing task stream */
  reconnectToTask: (
    sessionId: string,
    taskId: string,
    lastMessageId?: string,
    onChunk?: (chunk: StreamChunk) => void,
  ) => Promise<void>;
  /** Update last message ID for a task (for tracking replay position) */
  updateTaskLastMessageId: (sessionId: string, lastMessageId: string) => void;
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
    if (now - result.completedAt > COMPLETED_STREAM_TTL_MS) {
      cleaned.delete(sessionId);
    }
  }
  return cleaned;
}

/**
 * Finalize a stream by moving it from activeStreams to completedStreams.
 * Also handles cleanup and notifications.
 */
function finalizeStream(
  sessionId: string,
  stream: ActiveStream,
  onChunk: ((chunk: StreamChunk) => void) | undefined,
  get: () => ChatStoreState & ChatStoreActions,
  set: (state: Partial<ChatStoreState>) => void,
): void {
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
        notifyStreamComplete(currentState.streamCompleteCallbacks, sessionId);
      }
    }
  }
}

/**
 * Clean up an existing stream for a session and move it to completed streams.
 * Returns updated maps for both active and completed streams.
 */
function cleanupExistingStream(
  sessionId: string,
  activeStreams: Map<string, ActiveStream>,
  completedStreams: Map<string, StreamResult>,
  callbacks: Set<StreamCompleteCallback>,
): {
  activeStreams: Map<string, ActiveStream>;
  completedStreams: Map<string, StreamResult>;
} {
  const newActiveStreams = new Map(activeStreams);
  let newCompletedStreams = new Map(completedStreams);

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

  return {
    activeStreams: newActiveStreams,
    completedStreams: newCompletedStreams,
  };
}

/**
 * Create a new active stream with initial state.
 */
function createActiveStream(
  sessionId: string,
  onChunk?: (chunk: StreamChunk) => void,
): ActiveStream {
  const abortController = new AbortController();
  const initialCallbacks = new Set<(chunk: StreamChunk) => void>();
  if (onChunk) initialCallbacks.add(onChunk);

  return {
    sessionId,
    abortController,
    status: "streaming",
    startedAt: Date.now(),
    chunks: [],
    onChunkCallbacks: initialCallbacks,
  };
}

export const useChatStore = create<ChatStore>((set, get) => ({
  activeStreams: new Map(),
  completedStreams: new Map(),
  activeSessions: new Set(),
  streamCompleteCallbacks: new Set(),
  activeTasks: loadPersistedTasks(),

  startStream: async function startStream(
    sessionId,
    message,
    isUserMessage,
    context,
    onChunk,
  ) {
    const state = get();
    const callbacks = state.streamCompleteCallbacks;

    // Clean up any existing stream for this session
    const {
      activeStreams: newActiveStreams,
      completedStreams: newCompletedStreams,
    } = cleanupExistingStream(
      sessionId,
      state.activeStreams,
      state.completedStreams,
      callbacks,
    );

    // Create new stream
    const stream = createActiveStream(sessionId, onChunk);
    newActiveStreams.set(sessionId, stream);
    set({
      activeStreams: newActiveStreams,
      completedStreams: newCompletedStreams,
    });

    try {
      await executeStream(stream, message, isUserMessage, context);
    } finally {
      finalizeStream(sessionId, stream, onChunk, get, set);
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

  setActiveTask: function setActiveTask(sessionId, taskInfo) {
    const state = get();
    const newActiveTasks = new Map(state.activeTasks);
    newActiveTasks.set(sessionId, {
      ...taskInfo,
      sessionId,
      startedAt: Date.now(),
    });
    set({ activeTasks: newActiveTasks });
    persistTasks(newActiveTasks);
  },

  getActiveTask: function getActiveTask(sessionId) {
    return get().activeTasks.get(sessionId);
  },

  clearActiveTask: function clearActiveTask(sessionId) {
    const state = get();
    if (!state.activeTasks.has(sessionId)) return;

    const newActiveTasks = new Map(state.activeTasks);
    newActiveTasks.delete(sessionId);
    set({ activeTasks: newActiveTasks });
    persistTasks(newActiveTasks);
  },

  reconnectToTask: async function reconnectToTask(
    sessionId,
    taskId,
    lastMessageId = INITIAL_STREAM_ID,
    onChunk,
  ) {
    const state = get();
    const callbacks = state.streamCompleteCallbacks;

    // Clean up any existing stream for this session
    const {
      activeStreams: newActiveStreams,
      completedStreams: newCompletedStreams,
    } = cleanupExistingStream(
      sessionId,
      state.activeStreams,
      state.completedStreams,
      callbacks,
    );

    // Create new stream for reconnection
    const stream = createActiveStream(sessionId, onChunk);
    newActiveStreams.set(sessionId, stream);
    set({
      activeStreams: newActiveStreams,
      completedStreams: newCompletedStreams,
    });

    try {
      await executeTaskReconnect(stream, taskId, lastMessageId);
    } finally {
      finalizeStream(sessionId, stream, onChunk, get, set);

      // Clear active task on completion
      if (stream.status === "completed" || stream.status === "error") {
        const taskState = get();
        if (taskState.activeTasks.has(sessionId)) {
          const newActiveTasks = new Map(taskState.activeTasks);
          newActiveTasks.delete(sessionId);
          set({ activeTasks: newActiveTasks });
          persistTasks(newActiveTasks);
        }
      }
    }
  },

  updateTaskLastMessageId: function updateTaskLastMessageId(
    sessionId,
    lastMessageId,
  ) {
    const state = get();
    const task = state.activeTasks.get(sessionId);
    if (!task) return;

    const newActiveTasks = new Map(state.activeTasks);
    newActiveTasks.set(sessionId, {
      ...task,
      lastMessageId,
    });
    set({ activeTasks: newActiveTasks });
    persistTasks(newActiveTasks);
  },
}));
