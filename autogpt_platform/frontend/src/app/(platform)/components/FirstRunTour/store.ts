import { create } from "zustand";

// "auto": opened by itself on the first visit after onboarding.
// "menu": replayed from the profile menu's "Show me around".
export type TourSource = "auto" | "menu";

interface FirstRunTourState {
  source: TourSource | null;
  /** The step showing, by id; null is the first. */
  stepID: string | null;
  /** "Connect your tools", offered from the last step, outlives the tour. */
  isConnectToolsOpen: boolean;
  open: (source: TourSource) => void;
  goTo: (stepID: string) => void;
  close: () => void;
  setConnectToolsOpen: (isOpen: boolean) => void;
}

// Shared by the profile menu (replay), the home page (first visit) and the
// host that renders the walkthrough over the whole app.
export const useFirstRunTourStore = create<FirstRunTourState>((set) => ({
  source: null,
  stepID: null,
  isConnectToolsOpen: false,
  open(source) {
    set((state) => (state.source ? state : { source, stepID: null }));
  },
  goTo(stepID) {
    set({ stepID });
  },
  close() {
    set({ source: null, stepID: null });
  },
  setConnectToolsOpen(isConnectToolsOpen) {
    set({ isConnectToolsOpen });
  },
}));
