"use client";

import { useCopilotUIStore } from "@/app/(platform)/copilot/store";
import { useIsMobile } from "@/app/(platform)/copilot/useIsMobile";
import { DotDistortionShader } from "@/components/ui/dot-distortion-shader";
import {
  SidebarInset,
  SidebarProvider,
  SidebarTrigger,
} from "@/components/ui/sidebar";
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";
import dynamic from "next/dynamic";
import { CSSProperties } from "react";
import { TourChatHost } from "./TourChatHost";
import { TourScenarioChips } from "./components/TourScenarioChips/TourScenarioChips";
import { TourSidebar } from "./components/TourSidebar/TourSidebar";
import { buildTourArtifactRef } from "./helpers";
import { getTourScenario } from "./script/tourScenarios";
import { useTourStore } from "./tourStore";

const ArtifactPanel = dynamic(
  () =>
    import(
      "@/app/(platform)/copilot/components/ArtifactPanel/ArtifactPanel"
    ).then((m) => m.ArtifactPanel),
  { ssr: false },
);

function TourBackdrop() {
  return (
    <DotDistortionShader
      dotGap={14}
      dotSize={1}
      opacity={0.2}
      isStatic
      className="pointer-events-none absolute inset-0 !bg-transparent [&_canvas]:opacity-70"
    />
  );
}

export function TourCopilot() {
  const activeScenarioId = useTourStore((s) => s.activeScenarioId);
  const scenario = getTourScenario(activeScenarioId);
  const appShellEnabled = useGetFlag(Flag.TOUR_APP_SHELL);
  const isMobile = useIsMobile();
  const openArtifact = useCopilotUIStore((s) => s.openArtifact);
  const clearArtifactPreview = useCopilotUIStore((s) => s.clearArtifactPreview);

  if (!appShellEnabled) {
    return (
      <div className="relative flex h-dvh w-full flex-col overflow-hidden bg-[#fafafa]">
        <TourBackdrop />
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

  return (
    <SidebarProvider
      style={{ "--sidebar-width": "19rem" } as CSSProperties}
      className="h-dvh min-h-0"
    >
      <TourSidebar />
      <SidebarInset className="min-h-0 overflow-hidden bg-[#fafafa]">
        <div className="relative flex h-full min-h-0 w-full">
          <div className="relative flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden">
            <TourBackdrop />
            <div className="relative z-10 flex min-h-0 flex-1 flex-col">
              <div className="flex h-12 shrink-0 items-center px-3 md:h-4">
                <SidebarTrigger className="md:hidden" />
              </div>
              <TourChatHost
                key={scenario.id}
                sessionId={scenario.id}
                script={scenario.script}
                onComplete={() => openArtifact(buildTourArtifactRef(scenario))}
                onReset={clearArtifactPreview}
              />
            </div>
          </div>
          {!isMobile && <ArtifactPanel />}
        </div>
        {isMobile && <ArtifactPanel mobile />}
      </SidebarInset>
    </SidebarProvider>
  );
}
