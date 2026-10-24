import { create } from "zustand";

// "auto": opened by itself on the first visit after onboarding.
// "menu": replayed from the profile menu's "Show me around".
export type TourSource = "auto" | "menu";

interface FirstRunTourState {
  source: TourSource | null;
  stepIndex: number;
  /** "Connect your tools", offered from the last step, outlives the tour. */
  isConnectToolsOpen: boolean;
  open: (source: TourSource) => void;
  goTo: (stepIndex: number) => void;
  close: () => void;
  setConnectToolsOpen: (isOpen: boolean) => void;
}

// Shared by the profile menu (replay), the home page (first visit) and the
// host that renders the walkthrough over the whole app.
export const useFirstRunTourStore = create<FirstRunTourState>((set) => ({
  source: null,
  stepIndex: 0,
  isConnectToolsOpen: false,
  open(source) {
    set((state) => (state.source ? state : { source, stepIndex: 0 }));
  },
  goTo(stepIndex) {
    set({ stepIndex });
  },
  close() {
    set({ source: null, stepIndex: 0 });
  },
  setConnectToolsOpen(isConnectToolsOpen) {
    set({ isConnectToolsOpen });
  },
}));
