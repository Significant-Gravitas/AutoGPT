"use client";

import { useSidebar } from "@/components/molecules/Sidebar/Sidebar";
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

  // A collapsed sidebar is borrowed: opened without saving the preference,
  // and collapsed again when the walkthrough ends.
  // Once per walkthrough: collapsing it mid-way is the person's own choice.
  const sidebarLoan = useRef<"pending" | "not-needed" | "borrowed" | "taken">(
    "pending",
  );
  useEffect(() => {
    if (!isOpen || isMobile) return;
    if (sidebarLoan.current === "pending") {
      sidebarLoan.current = isSidebarOpen ? "not-needed" : "borrowed";
      if (!isSidebarOpen) setSidebarOpen(true, { persist: false });
    } else if (sidebarLoan.current === "borrowed" && !isSidebarOpen) {
      sidebarLoan.current = "taken";
    }
  }, [isOpen, isMobile, isSidebarOpen, setSidebarOpen]);
  // The setter changes identity as the sidebar opens; hand back on unmount.
  const giveSidebarBack = useStableCallback(() => {
    if (sidebarLoan.current !== "borrowed") return;
    setSidebarOpen(false, { persist: false });
    // A remount (StrictMode runs effects twice) borrows it afresh.
    sidebarLoan.current = "pending";
  });
  useEffect(() => giveSidebarBack, [giveSidebarBack]);

  // On a phone the scrim and focus trap block a tap on the menu, so the
  // walkthrough opens the sidebar sheet itself for the steps that live there.
  const wantsSheet = isOpen && isMobile && step?.inSidebar === true;
  useEffect(() => {
    if (isMobile && isOpen) setOpenMobile(wantsSheet);
  }, [isMobile, isOpen, wantsSheet, setOpenMobile]);
  useEffect(() => {
    if (isMobile) return () => setOpenMobile(false);
  }, [isMobile, setOpenMobile]);

  useEffect(() => {
    if (source) trackFirstRun("first_run_tour_shown", { opened_from: source });
  }, [source]);

  // The stop names its own step: it can be reported after the next one is up.
  function trackStep(stop: MeasuredStop) {
    if (!source) return;
    trackFirstRun("first_run_tour_step_viewed", {
      opened_from: source,
      step_id: stop.stepID,
      step_index: steps.findIndex((candidate) => candidate.id === stop.stepID),
      target: stop.target,
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
      { opened_from: source, step_id: step.id, step_index: safeIndex, via },
    );
    setFirstRunTourSeen(userID);
    completeStep("FIRST_RUN_TOUR");
    close();
  }

  function startHere() {
    finish("completed", "start_here");
    focusStartingPoint();
  }

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
