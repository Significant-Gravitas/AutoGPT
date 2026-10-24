"use client";

import { useSidebar } from "@/components/ui/sidebar";
import { useIsBillingEnabled } from "@/hooks/useIsBillingEnabled";
import { useAuth } from "@/lib/auth/hooks/useAuth";
import { useOnboarding } from "@/providers/onboarding/onboarding-provider";
import { trackFirstRun } from "@/services/onboarding/first-run-analytics";
import { useUsageExperience } from "@/services/usageExperience/useUsageExperience";
import { useStableCallback } from "@/hooks/useStableCallback";
import { useEffect, useRef } from "react";
import { setFirstRunTourSeen } from "./helpers";
import { focusComposer, focusStartingPoint } from "./focusStartingPoint";
import { findStepIndex, getAvailableSteps, getStepBody } from "./steps";
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
    setOpenMobile,
  } = useSidebar();
  const { experience } = useUsageExperience();
  const isBillingEnabled = useIsBillingEnabled();
  const source = useFirstRunTourStore((state) => state.source);
  const stepID = useFirstRunTourStore((state) => state.stepID);
  const goTo = useFirstRunTourStore((state) => state.goTo);
  const close = useFirstRunTourStore((state) => state.close);
  const setConnectToolsOpen = useFirstRunTourStore(
    (state) => state.setConnectToolsOpen,
  );

  const isOpen = source !== null;
  const steps = getAvailableSteps({ isBillingEnabled });
  const safeIndex = findStepIndex(steps, stepID);
  const step = isOpen ? steps[safeIndex] : null;
  const isLastStep = safeIndex === steps.length - 1;
  const context = { isTrial: experience.isActiveTrial };

  // Every step but the last points at the sidebar, so a collapsed one is
  // opened for the walkthrough. It is only borrowed: not saved as the
  // person's preference, and collapsed again once the walkthrough ends.
  const openedSidebar = useRef(false);
  useEffect(() => {
    if (!isOpen || isMobile || isSidebarOpen) return;
    openedSidebar.current = true;
    setSidebarOpen(true, { persist: false });
  }, [isOpen, isMobile, isSidebarOpen, setSidebarOpen]);
  // The setter changes with the sidebar's state, so the hand-back reads the
  // latest one and runs only when the walkthrough goes.
  const giveSidebarBack = useStableCallback(() => {
    if (openedSidebar.current) setSidebarOpen(false, { persist: false });
  });
  useEffect(() => giveSidebarBack, [giveSidebarBack]);

  // On a phone the sidebar is a sheet over the page. Pointing at its menu
  // button would ask for a tap the scrim and focus trap do not allow, so the
  // walkthrough opens the sheet itself for the steps that live there, and
  // puts it away for the rest and once it ends.
  const wantsSheet = isOpen && isMobile && step?.inSidebar === true;
  useEffect(() => {
    if (isMobile && isOpen) setOpenMobile(wantsSheet);
  }, [isMobile, isOpen, wantsSheet, setOpenMobile]);
  useEffect(() => {
    if (isMobile) return () => setOpenMobile(false);
  }, [isMobile, setOpenMobile]);

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
    next: () => (isLastStep ? startHere() : goTo(steps[safeIndex + 1].id)),
    back: () => goTo(steps[Math.max(safeIndex - 1, 0)].id),
    skip: () => {
      finish("skipped");
      focusComposer();
    },
    startHere,
    connectToolsFirst,
  };
}
