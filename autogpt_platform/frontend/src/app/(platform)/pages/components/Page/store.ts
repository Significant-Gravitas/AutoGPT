import { create } from "zustand";

interface FollowedRunState {
  /** Per page, the run each workflow shows once started from the page, by library id. */
  runs: Record<string, Record<string, string>>;
  follow(pageKey: string, libraryAgentID: string, runID: string): void;
}

// Outlives the page: the library view remounts it when a first run swaps its layout.
export const useFollowedRunStore = create<FollowedRunState>((set) => ({
  runs: {},
  follow(pageKey, libraryAgentID, runID) {
    set((state) => ({
      runs: {
        ...state.runs,
        [pageKey]: { ...state.runs[pageKey], [libraryAgentID]: runID },
      },
    }));
  },
}));

export function useFollowedRunIDs(pageKey: string) {
  const runs = useFollowedRunStore((s) => s.runs[pageKey]);
  return runs ? Object.values(runs) : [];
}
