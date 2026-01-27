"use client";

import { create } from "zustand";

interface CopilotStoreState {
  isStreaming: boolean;
  isSwitchingSession: boolean;
  isInterruptModalOpen: boolean;
  pendingAction: (() => void) | null;
  newChatHandler: (() => void) | null;
  selectSessionHandler: ((sessionId: string) => void) | null;
  selectSessionWithInterruptHandler: ((sessionId: string) => void) | null;
}

interface CopilotStoreActions {
  setIsStreaming: (isStreaming: boolean) => void;
  setIsSwitchingSession: (isSwitchingSession: boolean) => void;
  setNewChatHandler: (handler: (() => void) | null) => void;
  setSelectSessionHandler: (handler: ((sessionId: string) => void) | null) => void;
  setSelectSessionWithInterruptHandler: (handler: ((sessionId: string) => void) | null) => void;
  requestNewChat: () => void;
  requestSelectSession: (sessionId: string) => void;
  confirmInterrupt: () => void;
  cancelInterrupt: () => void;
}

type CopilotStore = CopilotStoreState & CopilotStoreActions;

export const useCopilotStore = create<CopilotStore>((set, get) => ({
  isStreaming: false,
  isSwitchingSession: false,
  isInterruptModalOpen: false,
  pendingAction: null,
  newChatHandler: null,
  selectSessionHandler: null,
  selectSessionWithInterruptHandler: null,

  setIsStreaming(isStreaming) {
    set({ isStreaming });
  },

  setIsSwitchingSession(isSwitchingSession) {
    set({ isSwitchingSession });
  },

  setNewChatHandler(handler) {
    set({ newChatHandler: handler });
  },

  setSelectSessionHandler(handler) {
    set({ selectSessionHandler: handler });
  },

  setSelectSessionWithInterruptHandler(handler) {
    set({ selectSessionWithInterruptHandler: handler });
  },

  requestNewChat() {
    const { isStreaming, newChatHandler } = get();
    if (isStreaming) {
      set({ isInterruptModalOpen: true, pendingAction: newChatHandler });
    } else if (newChatHandler) {
      newChatHandler();
    }
  },

  requestSelectSession(sessionId) {
    const { isStreaming, selectSessionHandler, selectSessionWithInterruptHandler } = get();
    if (isStreaming) {
      if (!selectSessionWithInterruptHandler) return;
      set({
        isInterruptModalOpen: true,
        pendingAction: () => selectSessionWithInterruptHandler(sessionId),
      });
    } else {
      if (!selectSessionHandler) return;
      selectSessionHandler(sessionId);
    }
  },

  confirmInterrupt() {
    const { pendingAction } = get();
    set({ isInterruptModalOpen: false, pendingAction: null });
    if (pendingAction) pendingAction();
  },

  cancelInterrupt() {
    set({ isInterruptModalOpen: false, pendingAction: null });
  },
}));
