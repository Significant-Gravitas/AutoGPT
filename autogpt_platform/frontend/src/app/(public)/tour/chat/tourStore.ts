import { create } from "zustand";
import { DEFAULT_SCENARIO_ID } from "./script/tourScenarios";

interface TourState {
  activeScenarioId: string;
  /** Bumped on every sidebar selection — re-keys the demo so clicking a
   * scenario (even the already-active one) always restarts it fresh. */
  runId: number;
  setActiveScenario: (id: string) => void;
}

export const useTourStore = create<TourState>((set) => ({
  activeScenarioId: DEFAULT_SCENARIO_ID,
  runId: 0,
  setActiveScenario: (id) =>
    set((state) => ({ activeScenarioId: id, runId: state.runId + 1 })),
}));
