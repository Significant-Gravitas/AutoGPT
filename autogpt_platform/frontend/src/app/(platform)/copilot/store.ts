import { Key, storage } from "@/services/storage/local-storage";
import { create } from "zustand";

export interface DeleteTarget {
  id: string;
  title: string | null | undefined;
}

interface CopilotUIState {
  sessionToDelete: DeleteTarget | null;
  setSessionToDelete: (target: DeleteTarget | null) => void;

  isDrawerOpen: boolean;
  setDrawerOpen: (open: boolean) => void;

  completedSessionIDs: Set<string>;
  addCompletedSession: (id: string) => void;
  clearCompletedSession: (id: string) => void;
  clearAllCompletedSessions: () => void;

  isSoundEnabled: boolean;
  toggleSound: () => void;
}

export const useCopilotUIStore = create<CopilotUIState>((set) => ({
  sessionToDelete: null,
  setSessionToDelete: (target) => set({ sessionToDelete: target }),

  isDrawerOpen: false,
  setDrawerOpen: (open) => set({ isDrawerOpen: open }),

  completedSessionIDs: new Set<string>(),
  addCompletedSession: (id) =>
    set((state) => {
      const next = new Set(state.completedSessionIDs);
      next.add(id);
      return { completedSessionIDs: next };
    }),
  clearCompletedSession: (id) =>
    set((state) => {
      const next = new Set(state.completedSessionIDs);
      next.delete(id);
      return { completedSessionIDs: next };
    }),
  clearAllCompletedSessions: () =>
    set({ completedSessionIDs: new Set<string>() }),

  isSoundEnabled: storage.get(Key.COPILOT_SOUND_ENABLED) !== "false",
  toggleSound: () =>
    set((state) => {
      const next = !state.isSoundEnabled;
      storage.set(Key.COPILOT_SOUND_ENABLED, String(next));
      return { isSoundEnabled: next };
    }),
}));
