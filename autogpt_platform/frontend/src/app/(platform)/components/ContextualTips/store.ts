import { onAccountChange } from "@/lib/auth/on-account-change";
import { create } from "zustand";
import type { TipID } from "./tips";

interface ContextualTipState {
  /** Moments that happened, oldest first, waiting for a turn on screen. */
  requested: TipID[];
  request: (id: TipID) => void;
  remove: (id: TipID) => void;
  /** Moves a tip with nothing on screen to point at behind the others. */
  postpone: (id: TipID) => void;
  /** Tips already on screen during this visit to the app (not persisted). */
  shown: TipID[];
  markShown: (id: TipID) => void;
  /** "Got it": out of the queue, and not to come back during this visit. */
  putAway: (id: TipID) => void;
}

export const useContextualTipStore = create<ContextualTipState>((set) => ({
  requested: [],
  shown: [],
  markShown(id) {
    set((state) =>
      state.shown.includes(id) ? state : { shown: [...state.shown, id] },
    );
  },
  putAway(id) {
    set((state) => ({
      requested: state.requested.filter((requested) => requested !== id),
      shown: state.shown.filter((shown) => shown !== id),
    }));
  },
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
  postpone(id) {
    set((state) =>
      state.requested.length < 2 || !state.requested.includes(id)
        ? state
        : {
            requested: [
              ...state.requested.filter((requested) => requested !== id),
              id,
            ],
          },
    );
  },
}));

export function requestTip(id: TipID) {
  useContextualTipStore.getState().request(id);
}

// Another account signing in on this tab starts with neither list.
onAccountChange(() =>
  useContextualTipStore.setState({ requested: [], shown: [] }),
);
