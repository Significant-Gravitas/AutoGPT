"use client";

import { DotDistortionShader } from "@/components/ui/dot-distortion-shader";
import { TourChatHost } from "./TourChatHost";
import { TourScenarioChips } from "./components/TourScenarioChips/TourScenarioChips";
import { getTourScenario } from "./script/tourScenarios";
import { useTourStore } from "./tourStore";

export function TourCopilot() {
  const activeScenarioId = useTourStore((s) => s.activeScenarioId);
  const scenario = getTourScenario(activeScenarioId);

  return (
    <div className="relative flex h-dvh w-full flex-col overflow-hidden bg-[#fafafa]">
      <DotDistortionShader
        dotGap={14}
        dotSize={1}
        opacity={0.2}
        isStatic
        className="pointer-events-none absolute inset-0 !bg-transparent [&_canvas]:opacity-70"
      />
      <div className="relative z-10 flex min-h-0 flex-1 flex-col">
        <div className="px-3 pb-2 pt-5">
          <TourScenarioChips />
        </div>
        <TourChatHost
          key={scenario.id}
          sessionId={scenario.id}
          script={scenario.script}
        />
      </div>
    </div>
  );
}
