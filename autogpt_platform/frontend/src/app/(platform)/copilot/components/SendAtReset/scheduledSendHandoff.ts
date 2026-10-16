import { create } from "zustand";
import { createJSONStorage, persist } from "zustand/middleware";

/**
 * "Send at reset" from the new-chat screen: the chat is created first, and
 * creating it remounts the page under the new session id. The message waits
 * here until that session's composer picks it up and sets the send. The
 * composer that took it has already cleared, so the wait is kept in the
 * tab's session storage: a reload in between still sets the send.
 */
interface ScheduledSendHandoff {
  messages: Record<string, string>;
  hold: (sessionId: string, message: string) => void;
  take: (sessionId: string) => string | null;
}

export const useScheduledSendHandoff = create<ScheduledSendHandoff>()(
  persist(
    (set, get) => ({
      messages: {},
      hold(sessionId, message) {
        set((state) => ({
          messages: { ...state.messages, [sessionId]: message },
        }));
      },
      take(sessionId) {
        const message = get().messages[sessionId] ?? null;
        if (message === null) return null;
        set((state) => {
          const rest = { ...state.messages };
          delete rest[sessionId];
          return { messages: rest };
        });
        return message;
      },
    }),
    {
      name: "copilot-scheduled-send-handoff",
      // No-op storage where there is no window (SSR), as in the stream store.
      storage: createJSONStorage(() =>
        typeof window !== "undefined" && window.sessionStorage
          ? window.sessionStorage
          : { getItem: () => null, setItem: () => {}, removeItem: () => {} },
      ),
      partialize: (state) => ({ messages: state.messages }),
    },
  ),
);
