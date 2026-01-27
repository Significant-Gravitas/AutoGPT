"use client";

import { create } from "zustand";

interface CopilotStoreState {
  isStreaming: boolean;
  isNewChatModalOpen: boolean;
  newChatHandler: (() => void) | null;
}

interface CopilotStoreActions {
  setIsStreaming: (isStreaming: boolean) => void;
  setNewChatHandler: (handler: (() => void) | null) => void;
  requestNewChat: () => void;
  confirmNewChat: () => void;
  cancelNewChat: () => void;
}

type CopilotStore = CopilotStoreState & CopilotStoreActions;

export const useCopilotStore = create<CopilotStore>((set, get) => ({
  isStreaming: false,
  isNewChatModalOpen: false,
  newChatHandler: null,

  setIsStreaming(isStreaming) {
    set({ isStreaming });
  },

  setNewChatHandler(handler) {
    set({ newChatHandler: handler });
  },

  requestNewChat() {
    const { isStreaming, newChatHandler } = get();
    if (isStreaming) {
      set({ isNewChatModalOpen: true });
    } else if (newChatHandler) {
      newChatHandler();
    }
  },

  confirmNewChat() {
    const { newChatHandler } = get();
    set({ isNewChatModalOpen: false });
    if (newChatHandler) {
      newChatHandler();
    }
  },

  cancelNewChat() {
    set({ isNewChatModalOpen: false });
  },
}));
