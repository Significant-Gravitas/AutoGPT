import type { UIMessage } from "ai";
import type { StoredAttachmentPart } from "./helpers/workspaceAttachments";
import { create } from "zustand";
import { createJSONStorage, persist } from "zustand/middleware";
import type { ExpertKickoffMetadata } from "./expertKickoff";

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
  lastSubmittedKickoffExpertId: string | null;
  lastSubmittedKickoffAttemptToken: string | null;
}

const defaultCoord: SessionCoord = {
  lastSubmittedMessageText: null,
  lastSubmittedKickoffExpertId: null,
  lastSubmittedKickoffAttemptToken: null,
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
  metadata?: ExpertKickoffMetadata;
}

export interface PendingUploadAttachment {
  name: string;
  mediaType: string;
  sizeBytes?: number;
  /** Local files upload before the send; workspace references are already stored. */
  isUploading: boolean;
}

/**
 * The message the user just sent, shown as a placeholder bubble while its
 * local attachments upload. The real user message only enters `messages`
 * once `sendMessage` runs after the uploads, which can take seconds for
 * large files — without this the transcript stays blank for that window.
 *
 * Kept per session in `pendingUploadSends` so an upload started in one chat
 * neither hides nor clears the placeholder of another chat the user switches
 * to mid-upload. The first send of a new chat has no session yet and sits
 * under `UNBOUND_UPLOAD_KEY` until `bindPendingFirstSendToSession` moves it
 * onto the created session. Lives here (not in React state) for the same
 * remount reason as `PendingFirstSend`. Never persisted: a reload mid-upload
 * loses the `File` objects, so there is nothing left to wait for.
 */
export interface PendingUploadSend {
  text: string;
  attachments: PendingUploadAttachment[];
}

/** Key for the first send of a chat whose session does not exist yet. */
const UNBOUND_UPLOAD_KEY = "new";

export function pendingUploadKey(sessionId: string | null): string {
  return sessionId ?? UNBOUND_UPLOAD_KEY;
}

interface PersistedCopilotStreamState {
  sessions: Record<string, SessionCoord>;
  pendingFirstSend: Pick<PendingFirstSend, "text" | "metadata"> | null;
  pendingFirstSendSessionId: string | null;
  pendingFileParts: StoredAttachmentPart[];
}

interface CopilotStreamStore {
  sessions: Record<string, SessionCoord>;
  messageSnapshots: Record<string, UIMessage[]>;
  pendingFirstSend: PendingFirstSend | null;
  pendingFirstSendSessionId: string | null;
  pendingFileParts: StoredAttachmentPart[];
  pendingUploadSends: Record<string, PendingUploadSend>;

  getCoord: (sessionId: string) => SessionCoord;
  updateCoord: (sessionId: string, patch: Partial<SessionCoord>) => void;
  clearSession: (sessionId: string) => void;
  getMessageSnapshot: (sessionId: string) => UIMessage[];
  setMessageSnapshot: (sessionId: string, messages: UIMessage[]) => void;

  setPendingFirstSend: (send: PendingFirstSend | null) => void;
  bindPendingFirstSendToSession: (sessionId: string) => void;
  setPendingFileParts: (parts: StoredAttachmentPart[]) => void;
  setPendingUploadSend: (
    sessionId: string | null,
    send: PendingUploadSend,
  ) => void;
  /** Clears the session's placeholder; with `send`, only if it is still the
   *  one shown, so a finished upload cannot clear a newer one. */
  clearPendingUploadSend: (
    sessionId: string | null,
    send?: PendingUploadSend,
  ) => void;
  /** Read-and-clear; used by the post-session-creation flush effect. */
  takePendingFirstSend: (sessionId: string) => {
    send: PendingFirstSend | null;
    parts: StoredAttachmentPart[];
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
      pendingUploadSends: {},

      getCoord(sessionId) {
        return { ...defaultCoord, ...get().sessions[sessionId] };
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
        set({ pendingFirstSend: send, pendingFirstSendSessionId: null });
      },
      bindPendingFirstSendToSession(sessionId) {
        if (!get().pendingFirstSend) return;
        set((state) => {
          const unbound = state.pendingUploadSends[UNBOUND_UPLOAD_KEY];
          if (!unbound) return { pendingFirstSendSessionId: sessionId };
          const pendingUploadSends = { ...state.pendingUploadSends };
          delete pendingUploadSends[UNBOUND_UPLOAD_KEY];
          pendingUploadSends[sessionId] = unbound;
          return { pendingFirstSendSessionId: sessionId, pendingUploadSends };
        });
      },
      setPendingFileParts(parts) {
        set({ pendingFileParts: parts });
      },
      setPendingUploadSend(sessionId, send) {
        set((state) => ({
          pendingUploadSends: {
            ...state.pendingUploadSends,
            [pendingUploadKey(sessionId)]: send,
          },
        }));
      },
      clearPendingUploadSend(sessionId, send) {
        set((state) => {
          const key = pendingUploadKey(sessionId);
          const current = state.pendingUploadSends[key];
          if (!current || (send && current !== send)) return {};
          const pendingUploadSends = { ...state.pendingUploadSends };
          delete pendingUploadSends[key];
          return { pendingUploadSends };
        });
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
          pendingUploadSends: {},
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
            ? {
                text: state.pendingFirstSend!.text,
                metadata: state.pendingFirstSend!.metadata,
              }
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
                send: {
                  text: persisted.pendingFirstSend.text,
                  files: [],
                  metadata: persisted.pendingFirstSend.metadata,
                },
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
