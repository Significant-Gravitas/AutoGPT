"use client";

import { cn } from "@/lib/utils";
import type { Icon } from "@phosphor-icons/react";
import {
  HeadsetIcon,
  MagnifyingGlassIcon,
  PhoneCallIcon,
  SunIcon,
} from "@phosphor-icons/react";
import { tourScenarios } from "../../script/tourScenarios";
import { useTourStore } from "../../tourStore";

const SCENARIO_ICONS: Record<string, Icon> = {
  "daily-brief": SunIcon,
  "call-prep": PhoneCallIcon,
  "competitor-watch": MagnifyingGlassIcon,
  "support-queue": HeadsetIcon,
};

export function TourScenarioChips() {
  const activeScenarioId = useTourStore((s) => s.activeScenarioId);
  const setActiveScenario = useTourStore((s) => s.setActiveScenario);

  return (
    <div className="flex flex-wrap items-center justify-center gap-2">
      {tourScenarios.map((scenario) => {
        const ChipIcon = SCENARIO_ICONS[scenario.id];
        const isActive = scenario.id === activeScenarioId;
        return (
          <button
            key={scenario.id}
            type="button"
            onClick={() => setActiveScenario(scenario.id)}
            aria-pressed={isActive}
            className={cn(
              "flex items-center gap-2 rounded-full border bg-white px-4 py-2 text-sm font-medium transition-colors",
              isActive
                ? "border-violet-400 bg-violet-50 text-violet-700"
                : "border-zinc-200 text-zinc-600 hover:border-zinc-300 hover:text-zinc-900",
            )}
          >
            {ChipIcon && <ChipIcon className="size-4 shrink-0" />}
            {scenario.label}
          </button>
        );
      })}
    </div>
  );
}
