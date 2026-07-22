"use client";

import { useCopilotUIStore } from "@/app/(platform)/copilot/store";
import { useTourStore } from "./tourStore";

/** Switching scenarios always closes the previous run's artifact panel —
 * shared by the sidebar sessions, the end card and the nudge chip. */
export function useTourScenarioSelection() {
  const setActiveScenario = useTourStore((s) => s.setActiveScenario);
  const closeArtifactPanel = useCopilotUIStore((s) => s.closeArtifactPanel);

  return function selectScenario(id: string) {
    // Close (not just clear) the panel: a completed demo leaves
    // artifactPanel.isOpen=true, which keeps the chat column dimmed at
    // opacity-50 behind it. persist:false keeps the tour from leaking
    // panel state into the real /copilot, same as TourCopilot's cleanup.
    closeArtifactPanel({ persist: false });
    setActiveScenario(id);
  };
}
