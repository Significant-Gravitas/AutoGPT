import type { FileUIPart, UIMessage } from "ai";
import { create } from "zustand";

/**
 * Per-session state. Zustand (not React refs) so it can survive events we
 * actually want to carry across visits — currently just `lastSubmittedMessageText`
 * for duplicate-POST protection. Everything transient per mount — resume flags,
 * reconnect counters, hydration gates — lives inside `useCopilotStream` as
 * React refs/state and resets naturally when the chat subtree remounts with
 * `key={sessionId}`.
 *
 * - lastSubmittedMessageText: blocks duplicate POSTs on resume. Meaningful
 *   across visits.
 *
 * (Previously also tracked `lastChunkId` as a cursor for incremental resume
 * via `?last_chunk_id=…`. That optimisation is unsafe with AI SDK v5's
 * `UIMessageStream` parser — it throws `UIMessageStreamError` on any
 * `*-delta` / `*-end` whose `*-start` predecessor is missing from its
 * parser-local `activeTextParts` / `activeReasoningParts` state, and a
 * cursor-based XREAD skips the envelope + `*-start` chunks at the top of the
 * turn. Every resume now replays from `0-0`; overlap is handled by
 * `deduplicateMessages` on the consumer side.)
 */
export interface SessionCoord {
  lastSubmittedMessageText: string | null;
}

const defaultCoord: SessionCoord = {
  lastSubmittedMessageText: null,
};

/**
 * Pending user input that must survive the `null → id` session-creation
 * remount.
 *
 * `CopilotPage` keys the chat subtree by `sessionId ?? "new"`, so the moment
 * `createSession` resolves and the URL gains a sessionId, React tears down
 * the `"new"`-keyed host (wiping per-mount refs) and mounts a fresh one.
 * The first send was fired before that remount and needs a place to live
 * until the new host can pick it up — React-local state can't because it
 * was just unmounted, hence this single module-scoped slot.
 *
 * Assumes a single `CopilotPage` instance per tab. Two concurrent instances
 * (parallel routes, split panes) would collide on this slot; the solution
 * today is "don't do that". If it ever becomes a real requirement, key by
 * a mount-stable outer id threaded from `CopilotPage` through a context.
 */
export interface PendingFirstSend {
  text: string;
  files: File[];
}

interface CopilotStreamStore {
  sessions: Record<string, SessionCoord>;
  messageSnapshots: Record<string, UIMessage[]>;
  pendingFirstSend: PendingFirstSend | null;
  pendingFileParts: FileUIPart[];

  getCoord: (sessionId: string) => SessionCoord;
  updateCoord: (sessionId: string, patch: Partial<SessionCoord>) => void;
  clearSession: (sessionId: string) => void;
  getMessageSnapshot: (sessionId: string) => UIMessage[];
  setMessageSnapshot: (sessionId: string, messages: UIMessage[]) => void;

  setPendingFirstSend: (send: PendingFirstSend | null) => void;
  setPendingFileParts: (parts: FileUIPart[]) => void;
  /** Read-and-clear; used by the post-session-creation flush effect. */
  takePendingFirstSend: () => {
    send: PendingFirstSend | null;
    parts: FileUIPart[];
  };

  /** Test-only: wipe all per-session state. */
  resetAll: () => void;
}

export const useCopilotStreamStore = create<CopilotStreamStore>((set, get) => ({
  sessions: {},
  messageSnapshots: {},
  pendingFirstSend: null,
  pendingFileParts: [],

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
  clearSession(sessionId) {
    set((state) => {
      const sessions = { ...state.sessions };
      delete sessions[sessionId];

      const messageSnapshots = { ...state.messageSnapshots };
      delete messageSnapshots[sessionId];

      return {
        sessions,
        messageSnapshots,
      };
    });
  },
  getMessageSnapshot(sessionId) {
    return get().messageSnapshots[sessionId] ?? [];
  },
  setMessageSnapshot(sessionId, messages) {
    set((state) => ({
      messageSnapshots: {
        ...state.messageSnapshots,
        [sessionId]: messages,
      },
    }));
  },

  setPendingFirstSend(send) {
    set({ pendingFirstSend: send });
  },
  setPendingFileParts(parts) {
    set({ pendingFileParts: parts });
  },
  takePendingFirstSend() {
    const { pendingFirstSend, pendingFileParts } = get();
    set({ pendingFirstSend: null, pendingFileParts: [] });
    return { send: pendingFirstSend, parts: pendingFileParts };
  },

  resetAll() {
    set({
      sessions: {},
      messageSnapshots: {},
      pendingFirstSend: null,
      pendingFileParts: [],
    });
  },
}));

export const DEFAULT_SESSION_COORD = defaultCoord;
