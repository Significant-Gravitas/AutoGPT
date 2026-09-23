"use client";

import { useCopilotUIStore } from "@/app/(platform)/copilot/store";
import { useIsMobile } from "@/app/(platform)/copilot/useIsMobile";
import { DotDistortionShader } from "@/components/ui/dot-distortion-shader";
import { SidebarInset, SidebarProvider } from "@/components/ui/sidebar";
import { useMountEffect } from "@/hooks/useMountEffect";
import dynamic from "next/dynamic";
import { CSSProperties } from "react";
import { TourChatHost } from "./TourChatHost";
import { TourChatHeader } from "./components/TourChatHeader/TourChatHeader";
import { TourSidebar } from "./components/TourSidebar/TourSidebar";
import { buildTourArtifactRef } from "./helpers";
import { getTourScenario } from "./script/tourScenarios";
import { useTourStore } from "./tourStore";
import { trackTourScenarioComplete, trackTourStart } from "./tracking";

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
  const runId = useTourStore((s) => s.runId);
  const setDemoComplete = useTourStore((s) => s.setDemoComplete);
  const scenario = getTourScenario(activeScenarioId);
  const isMobile = useIsMobile();
  const openArtifact = useCopilotUIStore((s) => s.openArtifact);
  const closeArtifactPanel = useCopilotUIStore((s) => s.closeArtifactPanel);

  // The tour shares the copilot UI store but must never leak panel state
  // into the real /copilot: opens skip the localStorage write
  // (persist: false) and unmount closes the panel in memory the same way.
  useMountEffect(() => {
    trackTourStart();
    return () => closeArtifactPanel({ persist: false });
  });

  const chatColumn = (
    // The chat used to fade back when the artifact panel opened, but the end
    // card (the demo's follow-up CTA) now appears at that exact moment — it
    // must stay fully legible, so the column keeps full opacity.
    <div className="relative flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden">
      <TourBackdrop />
      <div className="relative z-10 flex min-h-0 flex-1 flex-col">
        <TourChatHeader
          scenarioLabel={scenario.label}
          scenarioIcon={scenario.icon}
        />
        <div className="h-4 shrink-0" />
        <TourChatHost
          key={`${scenario.id}-${runId}`}
          sessionId={scenario.id}
          script={scenario.script}
          completionNotice={`Your **"${scenario.completionArtifact.filename}"** will appear in a moment on the right side.`}
          onComplete={() => {
            trackTourScenarioComplete(scenario.id);
            setDemoComplete();
            openArtifact(buildTourArtifactRef(scenario), { persist: false });
          }}
        />
      </div>
    </div>
  );

  // The copilot panel sizes itself from the persisted store width — on the
  // tour we pin it to 500px instead (the !important arbitrary variant wins
  // over the panel's inline style) without touching the visitor's storage.
  const artifactPanels = (
    <>
      {!isMobile && (
        <div className="contents [&_[data-artifact-panel]]:!w-[500px]">
          <ArtifactPanel />
        </div>
      )}
      {isMobile && <ArtifactPanel mobile />}
    </>
  );

  return (
    <SidebarProvider
      style={{ "--sidebar-width": "19rem" } as CSSProperties}
      className="h-dvh min-h-0"
    >
      <TourSidebar />
      <SidebarInset className="min-h-0 overflow-hidden bg-[#fafafa]">
        <div className="relative flex h-full min-h-0 w-full">
          {chatColumn}
          {artifactPanels}
        </div>
      </SidebarInset>
    </SidebarProvider>
  );
}
