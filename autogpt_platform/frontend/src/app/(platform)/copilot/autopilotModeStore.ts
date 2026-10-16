import type { StreamChatRequestAutopilotMode } from "@/app/api/__generated__/models/streamChatRequestAutopilotMode";
import { create } from "zustand";

export type AutopilotMode = NonNullable<StreamChatRequestAutopilotMode>;

export const DEFAULT_AUTOPILOT_MODE: AutopilotMode = "auto";

// A new chat has no session id until its first send creates one.
const NEW_CHAT = "";

interface AutopilotModeStore {
  /** Modes the user picked in this tab, by session id. Only the selector
   *  writes here, so with the flag off nothing is ever sent. */
  choices: Record<string, AutopilotMode>;
  choose: (sessionId: string | null, mode: AutopilotMode) => void;
  bindNewChatToSession: (sessionId: string) => void;
}

export const useAutopilotModeStore = create<AutopilotModeStore>((set) => ({
  choices: {},
  choose(sessionId, mode) {
    set((state) => ({
      choices: { ...state.choices, [sessionId ?? NEW_CHAT]: mode },
    }));
  },
  bindNewChatToSession(sessionId) {
    set((state) => {
      const mode = state.choices[NEW_CHAT];
      if (!mode) return state;
      const choices = { ...state.choices, [sessionId]: mode };
      delete choices[NEW_CHAT];
      return { choices };
    });
  },
}));

export function useAutopilotModeChoice(sessionId: string | null) {
  return useAutopilotModeStore((state) => state.choices[sessionId ?? NEW_CHAT]);
}

export function getAutopilotModeChoice(sessionId: string | null) {
  return useAutopilotModeStore.getState().choices[sessionId ?? NEW_CHAT];
}
