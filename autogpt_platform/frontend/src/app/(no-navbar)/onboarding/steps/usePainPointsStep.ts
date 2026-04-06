import { useState } from "react";
import { useOnboardingWizardStore } from "../store";

const MAX_SELECTIONS = 3;

const ROLE_TOP_PICKS: Record<string, string[]> = {
  "Founder/CEO": [
    "Finding leads",
    "Reports & data",
    "Email & outreach",
    "Scheduling",
  ],
  Operations: ["CRM & data entry", "Scheduling", "Reports & data"],
  "Sales/BD": ["Finding leads", "Email & outreach", "CRM & data entry"],
  Marketing: ["Social media", "Email & outreach", "Research"],
  "Product/PM": ["Research", "Reports & data", "Scheduling"],
  Engineering: ["Research", "Reports & data", "CRM & data entry"],
  "HR/People": ["Scheduling", "Email & outreach", "CRM & data entry"],
};

export function getTopPickIDs(role: string) {
  return ROLE_TOP_PICKS[role] ?? [];
}

export function usePainPointsStep() {
  const role = useOnboardingWizardStore((s) => s.role);
  const painPoints = useOnboardingWizardStore((s) => s.painPoints);
  const otherPainPoint = useOnboardingWizardStore((s) => s.otherPainPoint);
  const storeToggle = useOnboardingWizardStore((s) => s.togglePainPoint);
  const setOtherPainPoint = useOnboardingWizardStore(
    (s) => s.setOtherPainPoint,
  );
  const nextStep = useOnboardingWizardStore((s) => s.nextStep);
  const [shaking, setShaking] = useState(false);

  const topIDs = getTopPickIDs(role);
  const hasSomethingElse = painPoints.includes("Something else");
  const atLimit = painPoints.length >= MAX_SELECTIONS;
  const canContinue =
    painPoints.length > 0 &&
    (!hasSomethingElse || Boolean(otherPainPoint.trim()));

  function togglePainPoint(id: string) {
    const alreadySelected = painPoints.includes(id);
    if (!alreadySelected && atLimit) {
      setShaking(true);
      setTimeout(() => setShaking(false), 600);
      return;
    }
    storeToggle(id);
  }

  function handleLaunch() {
    if (canContinue) {
      nextStep();
    }
  }

  return {
    topIDs,
    painPoints,
    otherPainPoint,
    togglePainPoint,
    setOtherPainPoint,
    hasSomethingElse,
    atLimit,
    shaking,
    canContinue,
    handleLaunch,
  };
}
