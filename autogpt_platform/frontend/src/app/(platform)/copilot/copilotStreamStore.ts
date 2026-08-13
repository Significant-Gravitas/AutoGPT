import type { FileUIPart, UIMessage } from "ai";
import { create } from "zustand";
import { createJSONStorage, persist } from "zustand/middleware";

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

interface PersistedCopilotStreamState {
  sessions: Record<string, SessionCoord>;
  pendingFirstSend: Pick<PendingFirstSend, "text"> | null;
  pendingFirstSendSessionId: string | null;
  pendingFileParts: FileUIPart[];
}

interface CopilotStreamStore {
  sessions: Record<string, SessionCoord>;
  messageSnapshots: Record<string, UIMessage[]>;
  pendingFirstSend: PendingFirstSend | null;
  pendingFirstSendSessionId: string | null;
  pendingFileParts: FileUIPart[];
  /**
   * True while the current chat is in `streaming` or `submitted` state.
   * Shared so views outside the chat tree (e.g. the workspace sidebar's
   * Progress tab) can decide whether to render an "active" task list view
   * or fall back to an "idle/done" view without prop drilling.
   */
  isStreaming: boolean;

  getCoord: (sessionId: string) => SessionCoord;
  updateCoord: (sessionId: string, patch: Partial<SessionCoord>) => void;
  clearSession: (sessionId: string) => void;
  getMessageSnapshot: (sessionId: string) => UIMessage[];
  setMessageSnapshot: (sessionId: string, messages: UIMessage[]) => void;
  setStreaming: (streaming: boolean) => void;

  setPendingFirstSend: (send: PendingFirstSend | null) => void;
  bindPendingFirstSendToSession: (sessionId: string) => void;
  setPendingFileParts: (parts: FileUIPart[]) => void;
  /** Read-and-clear; used by the post-session-creation flush effect. */
  takePendingFirstSend: (sessionId: string) => {
    send: PendingFirstSend | null;
    parts: FileUIPart[];
  };

  /** Test-only: wipe all per-session state. */
  resetAll: () => void;
}

export const useCopilotStreamStore = create<CopilotStreamStore>()(
  persist<CopilotStreamStore, [], [], PersistedCopilotStreamState>(
    (set, get) => ({
      sessions: {},
      messageSnapshots: {},
      pendingFirstSend: null,
      pendingFirstSendSessionId: null,
      pendingFileParts: [],
      isStreaming: false,

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
      setStreaming(streaming) {
        if (get().isStreaming === streaming) return;
        set({ isStreaming: streaming });
      },

      setPendingFirstSend(send) {
        set({ pendingFirstSend: send, pendingFirstSendSessionId: null });
      },
      bindPendingFirstSendToSession(sessionId) {
        if (!get().pendingFirstSend) return;
        set({ pendingFirstSendSessionId: sessionId });
      },
      setPendingFileParts(parts) {
        set({ pendingFileParts: parts });
      },
      takePendingFirstSend(sessionId) {
        const {
          pendingFirstSend,
          pendingFirstSendSessionId,
          pendingFileParts,
        } = get();
        if (pendingFirstSendSessionId !== sessionId) {
          return { send: null, parts: [] };
        }
        set({
          pendingFirstSend: null,
          pendingFirstSendSessionId: null,
          pendingFileParts: [],
        });
        return { send: pendingFirstSend, parts: pendingFileParts };
      },

      resetAll() {
        set({
          sessions: {},
          messageSnapshots: {},
          pendingFirstSend: null,
          pendingFirstSendSessionId: null,
          pendingFileParts: [],
          isStreaming: false,
        });
      },
    }),
    {
      // Persist the per-session dedup memory plus the serializable portion of
      // the first send. A full navigation can happen while adding sessionId to
      // the URL on mobile, so a module-scoped slot alone is not sufficient.
      // Browser File objects remain memory-only; workspace FileUIParts are
      // already server-backed and safe to restore after navigation.
      // ``messageSnapshots`` is intentionally excluded — it's a per-render
      // cache of UIMessages (often hundreds) that the next mount should
      // re-derive from the server, not restore from storage.
      name: "copilot-stream-store",
      version: 1,
      // SSR-safe storage adapter: ``window.sessionStorage`` in the browser,
      // a no-op stub during Next.js SSR / vitest where ``window`` is
      // undefined.  Returning ``undefined`` from the factory would make
      // zustand throw on its first ``getItem`` call.
      storage: createJSONStorage(() =>
        typeof window !== "undefined" && window.sessionStorage
          ? window.sessionStorage
          : { getItem: () => null, setItem: () => {}, removeItem: () => {} },
      ),
      partialize(state) {
        const canRestorePending =
          state.pendingFirstSend !== null &&
          state.pendingFirstSend.files.length === 0 &&
          state.pendingFirstSendSessionId !== null;
        return {
          sessions: state.sessions,
          pendingFirstSend: canRestorePending
            ? { text: state.pendingFirstSend!.text }
            : null,
          pendingFirstSendSessionId: canRestorePending
            ? state.pendingFirstSendSessionId
            : null,
          pendingFileParts: canRestorePending ? state.pendingFileParts : [],
        };
      },
      merge(persistedState, currentState) {
        const persisted =
          persistedState as Partial<PersistedCopilotStreamState>;
        const restoredPending =
          persisted.pendingFirstSend &&
          typeof persisted.pendingFirstSendSessionId === "string"
            ? {
                send: { text: persisted.pendingFirstSend.text, files: [] },
                sessionId: persisted.pendingFirstSendSessionId,
                parts: persisted.pendingFileParts ?? [],
              }
            : null;
        return {
          ...currentState,
          sessions: persisted.sessions ?? {},
          pendingFirstSend: restoredPending?.send ?? null,
          pendingFirstSendSessionId: restoredPending?.sessionId ?? null,
          pendingFileParts: restoredPending?.parts ?? [],
        };
      },
    },
  ),
);

export const DEFAULT_SESSION_COORD = defaultCoord;
