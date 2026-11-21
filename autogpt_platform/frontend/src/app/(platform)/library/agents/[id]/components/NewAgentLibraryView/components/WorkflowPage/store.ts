import { create } from "zustand";

interface FollowedRunState {
  /** The run each workflow's page shows, by library id, once started from it. */
  runs: Record<string, string>;
  follow(libraryAgentID: string, runID: string): void;
}

// Outlives the page: a first run swaps the library view's layout and remounts it.
export const useFollowedRunStore = create<FollowedRunState>((set) => ({
  runs: {},
  follow(libraryAgentID, runID) {
    set((state) => ({ runs: { ...state.runs, [libraryAgentID]: runID } }));
  },
}));
