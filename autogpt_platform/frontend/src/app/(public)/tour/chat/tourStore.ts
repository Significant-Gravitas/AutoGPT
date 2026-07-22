import { create } from "zustand";
import { DEFAULT_SCENARIO_ID } from "./script/tourScenarios";

interface TourState {
  activeScenarioId: string;
  /** Bumped on every sidebar selection — re-keys the demo so clicking a
   * scenario (even the already-active one) always restarts it fresh. */
  runId: number;
  /** True once the active demo has played through — the end card takes over
   * the upsell and the sidebar card hides until a new scenario starts. */
  isDemoComplete: boolean;
  /** Scenarios the visitor has watched to completion — the sidebar marks
   * them with a check and the "next scenario" nudge skips them. */
  watchedScenarioIds: string[];
  /** True once the visitor has idled past the end card — shows the
   * "Next: …" chip and pulses the next scenario in the sidebar. */
  isNudgeVisible: boolean;
  setActiveScenario: (id: string) => void;
  setDemoComplete: () => void;
  showNudge: () => void;
}

export const useTourStore = create<TourState>((set) => ({
  activeScenarioId: DEFAULT_SCENARIO_ID,
  runId: 0,
  isDemoComplete: false,
  watchedScenarioIds: [],
  isNudgeVisible: false,
  setActiveScenario: (id) =>
    set((state) => ({
      activeScenarioId: id,
      runId: state.runId + 1,
      isDemoComplete: false,
      isNudgeVisible: false,
    })),
  setDemoComplete: () =>
    set((state) => ({
      isDemoComplete: true,
      watchedScenarioIds: state.watchedScenarioIds.includes(
        state.activeScenarioId,
      )
        ? state.watchedScenarioIds
        : [...state.watchedScenarioIds, state.activeScenarioId],
    })),
  showNudge: () => set({ isNudgeVisible: true }),
}));
