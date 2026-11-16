import { onAccountChange } from "@/lib/auth/on-account-change";
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
  /** "Show me around" from another page: opens once Home has loaded. */
  pendingReplay: boolean;
  requestReplay: () => void;
  startPendingReplay: () => void;
  open: (source: TourSource) => void;
  goTo: (stepID: string) => void;
  close: () => void;
  setConnectToolsOpen: (isOpen: boolean) => void;
}

export const useFirstRunTourStore = create<FirstRunTourState>((set) => ({
  source: null,
  stepID: null,
  isConnectToolsOpen: false,
  pendingReplay: false,
  requestReplay() {
    set({ pendingReplay: true });
  },
  startPendingReplay() {
    set((state) =>
      !state.pendingReplay
        ? state
        : state.source
          ? { pendingReplay: false }
          : { pendingReplay: false, source: "menu", stepID: null },
    );
  },
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

// Another account signing in on this tab starts with it closed.
onAccountChange(() =>
  useFirstRunTourStore.setState({
    source: null,
    stepID: null,
    isConnectToolsOpen: false,
    pendingReplay: false,
  }),
);
