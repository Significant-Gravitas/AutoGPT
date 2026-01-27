"use client";

import { create } from "zustand";

interface CopilotStoreState {
  isStreaming: boolean;
  isSwitchingSession: boolean;
  isInterruptModalOpen: boolean;
  pendingAction: (() => void) | null;
}

interface CopilotStoreActions {
  setIsStreaming: (isStreaming: boolean) => void;
  setIsSwitchingSession: (isSwitchingSession: boolean) => void;
  openInterruptModal: (onConfirm: () => void) => void;
  confirmInterrupt: () => void;
  cancelInterrupt: () => void;
}

type CopilotStore = CopilotStoreState & CopilotStoreActions;

export const useCopilotStore = create<CopilotStore>((set, get) => ({
  isStreaming: false,
  isSwitchingSession: false,
  isInterruptModalOpen: false,
  pendingAction: null,

  setIsStreaming(isStreaming) {
    set({ isStreaming });
  },

  setIsSwitchingSession(isSwitchingSession) {
    set({ isSwitchingSession });
  },

  openInterruptModal(onConfirm) {
    set({ isInterruptModalOpen: true, pendingAction: onConfirm });
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
