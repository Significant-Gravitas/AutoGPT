"use client";

import { create } from "zustand";

interface CopilotStoreState {
  isStreaming: boolean;
  isCreatingSession: boolean;
}

interface CopilotStoreActions {
  setIsStreaming: (isStreaming: boolean) => void;
  setIsCreatingSession: (isCreating: boolean) => void;
}

type CopilotStore = CopilotStoreState & CopilotStoreActions;

export const useCopilotStore = create<CopilotStore>((set) => ({
  isStreaming: false,
  isCreatingSession: false,

  setIsStreaming(isStreaming) {
    set({ isStreaming });
  },

  setIsCreatingSession(isCreatingSession) {
    set({ isCreatingSession });
  },
}));
