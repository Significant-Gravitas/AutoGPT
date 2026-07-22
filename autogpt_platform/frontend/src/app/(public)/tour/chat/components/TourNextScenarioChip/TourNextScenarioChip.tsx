"use client";

import { PlayIcon } from "@phosphor-icons/react";
import { TOUR_NEXT_SCENARIO_SECONDS } from "../../constants";
import { getNextTourScenario } from "../../script/tourScenarios";
import { useTourStore } from "../../tourStore";
import { trackTourCtaClick } from "../../tracking";
import { useTourScenarioSelection } from "../../useTourScenarioSelection";

export function TourNextScenarioChip() {
  const activeScenarioId = useTourStore((s) => s.activeScenarioId);
  const watchedScenarioIds = useTourStore((s) => s.watchedScenarioIds);
  const selectScenario = useTourScenarioSelection();

  const nextScenario = getNextTourScenario(
    activeScenarioId,
    watchedScenarioIds,
  );

  function handleClick() {
    trackTourCtaClick("another-scenario", { placement: "nudge-chip" });
    selectScenario(nextScenario.id);
  }

  return (
    <div className="flex justify-center pb-2">
      <button
        type="button"
        onClick={handleClick}
        className="inline-flex items-center gap-1.5 rounded-full border border-dashed border-violet-400 bg-violet-50/80 px-3.5 py-1.5 text-sm font-medium text-violet-700 transition-colors duration-300 animate-in fade-in slide-in-from-bottom-1 hover:bg-violet-100"
      >
        <PlayIcon className="size-3.5 shrink-0" weight="fill" />
        <span className="truncate">
          Next: {nextScenario.label} — watch it in {TOUR_NEXT_SCENARIO_SECONDS}s
        </span>
      </button>
    </div>
  );
}
