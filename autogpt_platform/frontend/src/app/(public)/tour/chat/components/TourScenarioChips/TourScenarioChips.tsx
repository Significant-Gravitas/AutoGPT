"use client";

import { useCopilotUIStore } from "@/app/(platform)/copilot/store";
import { cn } from "@/lib/utils";
import { tourScenarios } from "../../script/tourScenarios";
import { useTourStore } from "../../tourStore";

export function TourScenarioChips() {
  const activeScenarioId = useTourStore((s) => s.activeScenarioId);
  const setActiveScenario = useTourStore((s) => s.setActiveScenario);
  const clearArtifactPreview = useCopilotUIStore((s) => s.clearArtifactPreview);

  function selectScenario(id: string) {
    clearArtifactPreview();
    setActiveScenario(id);
  }

  return (
    <div className="flex flex-wrap items-center justify-center gap-2">
      {tourScenarios.map((scenario) => {
        const ChipIcon = scenario.icon;
        const isActive = scenario.id === activeScenarioId;
        return (
          <button
            key={scenario.id}
            type="button"
            onClick={() => selectScenario(scenario.id)}
            aria-pressed={isActive}
            className={cn(
              "flex items-center gap-2 rounded-full border bg-white px-4 py-2 text-sm font-medium transition-colors",
              isActive
                ? "border-violet-400 bg-violet-50 text-violet-700"
                : "border-zinc-200 text-zinc-600 hover:border-zinc-300 hover:text-zinc-900",
            )}
          >
            <ChipIcon className="size-4 shrink-0" />
            {scenario.label}
          </button>
        );
      })}
    </div>
  );
}
