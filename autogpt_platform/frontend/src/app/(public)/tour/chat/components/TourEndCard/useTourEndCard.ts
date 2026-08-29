"use client";

import { useMountEffect } from "@/hooks/useMountEffect";
import { getNextTourScenario } from "../../script/tourScenarios";
import { useTourStore } from "../../tourStore";
import { trackTourCtaClick } from "../../tracking";
import { useTourScenarioSelection } from "../../useTourScenarioSelection";

/** How long the visitor can sit on the finished demo before the "Next: …"
 * chip appears and the next sidebar scenario starts pulsing. */
const IDLE_NUDGE_DELAY_MS = 4000;

export function useTourEndCard() {
  const activeScenarioId = useTourStore((s) => s.activeScenarioId);
  const watchedScenarioIds = useTourStore((s) => s.watchedScenarioIds);
  const showNudge = useTourStore((s) => s.showNudge);
  const selectScenario = useTourScenarioSelection();

  const nextScenario = getNextTourScenario(
    activeScenarioId,
    watchedScenarioIds,
  );

  // The card only mounts once the demo completes, so idle time starts here.
  useMountEffect(() => {
    const timer = setTimeout(showNudge, IDLE_NUDGE_DELAY_MS);
    return () => clearTimeout(timer);
  });

  function handlePricingClick() {
    trackTourCtaClick("pricing", { placement: "end-card" });
  }

  function handleSelfHostClick() {
    trackTourCtaClick("self-host", { placement: "end-card" });
  }

  function handleWatchAnother() {
    trackTourCtaClick("another-scenario", { placement: "end-card" });
    selectScenario(nextScenario.id);
  }

  return {
    handlePricingClick,
    handleSelfHostClick,
    handleWatchAnother,
  };
}
