import { create } from "zustand";
import { createJSONStorage, persist } from "zustand/middleware";

/**
 * "Send at reset" from the new-chat screen: the chat is created first, and
 * creating it remounts the page under the new session id. The message waits
 * here until that session's composer has set the send. The composer that took
 * it has already cleared, so the wait is kept in the tab's session storage
 * and only ends once the request that sets the send has settled: a reload
 * before then, or while it's in flight, still finds the message.
 */
interface HandedOffSend {
  message: string;
  /** When a page sent it to be set; null until then. */
  sentAt: number | null;
}

interface ScheduledSendHandoff {
  sends: Record<string, HandedOffSend>;
  hold: (sessionId: string, message: string) => void;
  markSent: (sessionId: string) => void;
  clear: (sessionId: string) => void;
}

export const useScheduledSendHandoff = create<ScheduledSendHandoff>()(
  persist(
    (set) => ({
      sends: {},
      hold(sessionId, message) {
        set((state) => ({
          sends: { ...state.sends, [sessionId]: { message, sentAt: null } },
        }));
      },
      markSent(sessionId) {
        set((state) => {
          const send = state.sends[sessionId];
          if (!send) return state;
          return {
            sends: {
              ...state.sends,
              [sessionId]: { ...send, sentAt: Date.now() },
            },
          };
        });
      },
      clear(sessionId) {
        set((state) => {
          const rest = { ...state.sends };
          delete rest[sessionId];
          return { sends: rest };
        });
      },
    }),
    {
      name: "copilot-scheduled-send-handoff",
      version: 1,
      // Version 0 kept bare messages that no page had sent yet.
      migrate: (persisted) => ({ sends: migrateSends(persisted) }),
      // No-op storage where there is no window (SSR), as in the stream store.
      storage: createJSONStorage(() =>
        typeof window !== "undefined" && window.sessionStorage
          ? window.sessionStorage
          : { getItem: () => null, setItem: () => {}, removeItem: () => {} },
      ),
      partialize: (state) => ({ sends: state.sends }),
    },
  ),
);

function migrateSends(persisted: unknown): Record<string, HandedOffSend> {
  const messages =
    persisted && typeof persisted === "object" && "messages" in persisted
      ? (persisted as { messages: unknown }).messages
      : null;
  if (!messages || typeof messages !== "object") return {};
  return Object.fromEntries(
    Object.entries(messages)
      .filter(
        (entry): entry is [string, string] => typeof entry[1] === "string",
      )
      .map(([sessionId, message]) => [sessionId, { message, sentAt: null }]),
  );
}
