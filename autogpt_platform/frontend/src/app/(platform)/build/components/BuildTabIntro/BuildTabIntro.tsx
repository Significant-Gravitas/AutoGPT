"use client";

import { TabIntroCard } from "@/app/(platform)/components/TabIntroCard/TabIntroCard";
import { useTabIntroCard } from "@/app/(platform)/components/TabIntroCard/useTabIntroCard";
import { useTutorialStore } from "@/app/(platform)/build/stores/tutorialStore";
import { FlowIcon } from "@hugeicons/core-free-icons";
import { useRouter, useSearchParams } from "next/navigation";
import { startTutorial } from "../FlowEditor/tutorial";

// First visit to the Build tab. AutoPilot is the primary route on purpose —
// most users should not start on an empty canvas — with the existing builder
// tutorial kept as the quiet alternative for people who want the canvas.
export function BuildTabIntro() {
  // Deep link into an existing graph: the intro is about starting something,
  // and its tutorial clears the canvas — neither belongs over someone's saved
  // agent. Vetoing the visit leaves the step unrecorded, so a later blank
  // /build still gets the introduction.
  const isEditingSavedGraph = Boolean(useSearchParams().get("flowID"));
  const { isOpen, dismiss, takeAction } = useTabIntroCard(
    "build",
    !isEditingSavedGraph,
  );
  const setIsTutorialRunning = useTutorialStore(
    (state) => state.setIsTutorialRunning,
  );
  const router = useRouter();

  function askAutoPilot() {
    takeAction("ask_autopilot");
    router.push("/copilot");
  }

  function learnToBuild() {
    takeAction("builder_tutorial");
    startTutorial();
    setIsTutorialRunning(true);
  }

  return (
    <TabIntroCard
      isOpen={isOpen}
      icon={FlowIcon}
      title="Create your own workflows."
      body="Wire blocks into an agent that runs exactly how you want — or let AutoPilot build it."
      cta={{ label: "Ask AutoPilot to build it", onClick: askAutoPilot }}
      altAction={{ label: "Learn to build it yourself", onClick: learnToBuild }}
      onDismiss={dismiss}
    />
  );
}
