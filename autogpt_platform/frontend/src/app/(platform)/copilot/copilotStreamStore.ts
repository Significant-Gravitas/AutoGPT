import { create } from "zustand";

/**
 * Per-session state that must survive session-switches (hence Zustand, not
 * React refs). Everything transient per mount — resume flags, reconnect
 * counters, hydration gates — lives inside `useCopilotStream` as React
 * refs/state and resets naturally when the chat subtree remounts with
 * `key={sessionId}`.
 *
 * Fields below reflect server-side truth that's meaningful across visits:
 * - lastChunkId: cursor for incremental resume (XREAD exclusive successor)
 * - lastSubmittedMessageText: blocks duplicate POSTs on resume
 */
export interface SessionCoord {
  lastSubmittedMessageText: string | null;
  lastChunkId: string | null;
}

const defaultCoord: SessionCoord = {
  lastSubmittedMessageText: null,
  lastChunkId: null,
};

interface CopilotStreamStore {
  sessions: Record<string, SessionCoord>;

  getCoord: (sessionId: string) => SessionCoord;
  updateCoord: (sessionId: string, patch: Partial<SessionCoord>) => void;
  /** Test-only: wipe all per-session state. */
  resetAll: () => void;
}

export const useCopilotStreamStore = create<CopilotStreamStore>((set, get) => ({
  sessions: {},

  getCoord(sessionId) {
    return get().sessions[sessionId] ?? defaultCoord;
  },
  updateCoord(sessionId, patch) {
    set((state) => ({
      sessions: {
        ...state.sessions,
        [sessionId]: {
          ...(state.sessions[sessionId] ?? defaultCoord),
          ...patch,
        },
      },
    }));
  },
  resetAll() {
    set({ sessions: {} });
  },
}));

export const DEFAULT_SESSION_COORD = defaultCoord;
