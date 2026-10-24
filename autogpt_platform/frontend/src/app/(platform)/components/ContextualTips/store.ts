import { create } from "zustand";
import type { TipID } from "./tips";

interface ContextualTipState {
  /** Moments that happened, oldest first, waiting for a turn on screen. */
  requested: TipID[];
  request: (id: TipID) => void;
  remove: (id: TipID) => void;
}

// Surfaces report their moment here (a second chat, a cost, an expert
// proposal); the host decides when, and whether, the tip shows.
export const useContextualTipStore = create<ContextualTipState>((set) => ({
  requested: [],
  request(id) {
    set((state) =>
      state.requested.includes(id)
        ? state
        : { requested: [...state.requested, id] },
    );
  },
  remove(id) {
    set((state) => ({
      requested: state.requested.filter((requested) => requested !== id),
    }));
  },
}));

export function requestTip(id: TipID) {
  useContextualTipStore.getState().request(id);
}
