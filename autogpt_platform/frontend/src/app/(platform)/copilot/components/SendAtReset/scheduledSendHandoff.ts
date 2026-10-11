import { create } from "zustand";

/**
 * "Send at reset" from the new-chat screen: the chat is created first, and
 * creating it remounts the page under the new session id. The message waits
 * here until that session's composer picks it up and sets the send.
 */
interface ScheduledSendHandoff {
  messages: Record<string, string>;
  hold: (sessionId: string, message: string) => void;
  take: (sessionId: string) => string | null;
}

export const useScheduledSendHandoff = create<ScheduledSendHandoff>(
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
);
