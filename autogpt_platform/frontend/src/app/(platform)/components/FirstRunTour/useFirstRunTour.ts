"use client";

import { useSidebar } from "@/components/ui/sidebar";
import { useIsBillingEnabled } from "@/hooks/useIsBillingEnabled";
import { useAuth } from "@/lib/auth/hooks/useAuth";
import { useOnboarding } from "@/providers/onboarding/onboarding-provider";
import { trackFirstRun } from "@/services/onboarding/first-run-analytics";
import { useUsageExperience } from "@/services/usageExperience/useUsageExperience";
import { useEffect } from "react";
import { setFirstRunTourSeen } from "./helpers";
import { focusStartingPoint } from "./focusStartingPoint";
import { getAvailableSteps, getStepBody } from "./steps";
import { useFirstRunTourStore } from "./store";
import type { MeasuredStop } from "./useTourStops";

// The walkthrough's lifecycle, shared by every way of presenting it: which
// step is up, how it ends, and what ending it records.
export function useFirstRunTour() {
  const { user } = useAuth();
  const userID = user?.id ?? null;
  const { completeStep } = useOnboarding();
  const {
    isMobile,
    open: isSidebarOpen,
    setOpen: setSidebarOpen,
  } = useSidebar();
  const { experience } = useUsageExperience();
  const isBillingEnabled = useIsBillingEnabled();
  const source = useFirstRunTourStore((state) => state.source);
  const stepIndex = useFirstRunTourStore((state) => state.stepIndex);
  const goTo = useFirstRunTourStore((state) => state.goTo);
  const close = useFirstRunTourStore((state) => state.close);
  const setConnectToolsOpen = useFirstRunTourStore(
    (state) => state.setConnectToolsOpen,
  );

  const isOpen = source !== null;
  const steps = getAvailableSteps({ isBillingEnabled });
  const safeIndex = Math.min(stepIndex, steps.length - 1);
  const step = isOpen ? steps[safeIndex] : null;
  const isLastStep = safeIndex === steps.length - 1;
  const context = { isTrial: experience.isActiveTrial };

  // Every step but the last points at the sidebar. A new account starts with
  // it collapsed to icons, where the wallet is not shown at all, so open it
  // for the walkthrough and leave it open: that is where these things live.
  useEffect(() => {
    if (isOpen && !isMobile && !isSidebarOpen) setSidebarOpen(true);
  }, [isOpen, isMobile, isSidebarOpen, setSidebarOpen]);

  useEffect(() => {
    if (source) trackFirstRun("first_run_tour_shown", { source });
  }, [source]);

  function trackStep(stop: MeasuredStop | undefined) {
    if (!source || !step) return;
    trackFirstRun("first_run_tour_step_viewed", {
      source,
      step_id: step.id,
      step_index: safeIndex,
      target: stop?.target ?? null,
    });
  }

  function finish(
    outcome: "completed" | "skipped",
    via: "start_here" | "connect_tools" | null = null,
  ) {
    if (!source || !step) return;
    trackFirstRun(
      outcome === "completed"
        ? "first_run_tour_completed"
        : "first_run_tour_skipped",
      { source, step_id: step.id, step_index: safeIndex, via },
    );
    setFirstRunTourSeen(userID);
    completeStep("FIRST_RUN_TOUR");
    close();
  }

  function startHere() {
    finish("completed", "start_here");
    focusStartingPoint();
  }

  // The old first-run deck carried a "Connect your tools" picker, seeded
  // from the brain dump; the walkthrough keeps it one click from the end.
  function connectToolsFirst() {
    finish("completed", "connect_tools");
    setConnectToolsOpen(true);
  }

  return {
    steps,
    stepIndex: safeIndex,
    isLastStep,
    trackStep,
    bodyFor: (index: number, stop: MeasuredStop | undefined) =>
      getStepBody(steps[index], stop?.target ?? null, context),
    next: () => (isLastStep ? startHere() : goTo(safeIndex + 1)),
    back: () => goTo(Math.max(safeIndex - 1, 0)),
    skip: () => finish("skipped"),
    startHere,
    connectToolsFirst,
  };
}
