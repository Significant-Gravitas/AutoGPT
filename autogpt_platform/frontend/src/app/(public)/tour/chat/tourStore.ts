import { create } from "zustand";
import { DEFAULT_SCENARIO_ID } from "./script/tourScenarios";

interface TourState {
  activeScenarioId: string;
  setActiveScenario: (id: string) => void;
}

export const useTourStore = create<TourState>((set) => ({
  activeScenarioId: DEFAULT_SCENARIO_ID,
  setActiveScenario: (id) => set({ activeScenarioId: id }),
}));
