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
}

export const useCopilotUIStore = create<CopilotUIState>((set) => ({
  sessionToDelete: null,
  setSessionToDelete: (target) => set({ sessionToDelete: target }),

  isDrawerOpen: false,
  setDrawerOpen: (open) => set({ isDrawerOpen: open }),
}));
