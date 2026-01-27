"use client";

import { create } from "zustand";

interface CopilotStoreState {
  isStreaming: boolean;
  isSwitchingSession: boolean;
  isCreatingSession: boolean;
  isInterruptModalOpen: boolean;
  pendingAction: (() => void) | null;
}

interface CopilotStoreActions {
  setIsStreaming: (isStreaming: boolean) => void;
  setIsSwitchingSession: (isSwitchingSession: boolean) => void;
  setIsCreatingSession: (isCreating: boolean) => void;
  openInterruptModal: (onConfirm: () => void) => void;
  confirmInterrupt: () => void;
  cancelInterrupt: () => void;
}

type CopilotStore = CopilotStoreState & CopilotStoreActions;

export const useCopilotStore = create<CopilotStore>((set, get) => ({
  isStreaming: false,
  isSwitchingSession: false,
  isCreatingSession: false,
  isInterruptModalOpen: false,
  pendingAction: null,

  setIsStreaming(isStreaming) {
    set({ isStreaming });
  },

  setIsSwitchingSession(isSwitchingSession) {
    set({ isSwitchingSession });
  },

  setIsCreatingSession(isCreatingSession) {
    set({ isCreatingSession });
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
