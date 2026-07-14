import { create } from "zustand";
import { DEFAULT_SCENARIO_ID } from "./script/tourScenarios";

interface TourState {
  activeScenarioId: string;
  /** Bumped on every sidebar selection — re-keys the demo so clicking a
   * scenario (even the already-active one) always restarts it fresh. */
  runId: number;
  /** True once the active demo has played through — the sidebar upsell card
   * saves its attention-grabbing animations for this moment. */
  isDemoComplete: boolean;
  setActiveScenario: (id: string) => void;
  setDemoComplete: () => void;
}

export const useTourStore = create<TourState>((set) => ({
  activeScenarioId: DEFAULT_SCENARIO_ID,
  runId: 0,
  isDemoComplete: false,
  setActiveScenario: (id) =>
    set((state) => ({
      activeScenarioId: id,
      runId: state.runId + 1,
      isDemoComplete: false,
    })),
  setDemoComplete: () => set({ isDemoComplete: true }),
}));
