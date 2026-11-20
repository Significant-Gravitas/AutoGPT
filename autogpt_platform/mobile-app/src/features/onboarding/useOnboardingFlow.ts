import * as Crypto from "expo-crypto";
import { useState } from "react";

import type { OnboardingStepKey, WorkAnswer } from "./helpers";

/** The three steps' answers. Nothing is saved until setup; quitting starts again. */
export function useOnboardingFlow() {
  const [step, setStep] = useState<OnboardingStepKey>("role");
  const [role, setRole] = useState("");
  const [otherRole, setOtherRole] = useState("");
  const [work, setWork] = useState<WorkAnswer | null>(null);
  // One per run, so a retried setup finalizes the same brain dump.
  const [recordingID] = useState(() => Crypto.randomUUID());

  function pickRole(id: string) {
    setRole(id);
  }

  function goToWork() {
    setStep("work");
  }

  function backToRole() {
    setStep("role");
  }

  function submitWork(answer: WorkAnswer) {
    setWork(answer);
    setStep("setup");
  }

  function showPlanNeeded() {
    setStep("plan");
  }

  return {
    step,
    role,
    otherRole,
    work,
    recordingID,
    pickRole,
    setOtherRole,
    goToWork,
    backToRole,
    submitWork,
    showPlanNeeded,
  };
}
