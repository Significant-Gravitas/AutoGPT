"use client";

import { useCopilotUIStore } from "@/app/(platform)/copilot/store";
import { useTourStore } from "./tourStore";

/** Switching scenarios always clears the previous run's artifact preview —
 * shared by the sidebar sessions, the end card and the nudge chip. */
export function useTourScenarioSelection() {
  const setActiveScenario = useTourStore((s) => s.setActiveScenario);
  const clearArtifactPreview = useCopilotUIStore((s) => s.clearArtifactPreview);

  return function selectScenario(id: string) {
    clearArtifactPreview();
    setActiveScenario(id);
  };
}
