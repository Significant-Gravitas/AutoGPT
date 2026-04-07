import { useOnboardingWizardStore } from "../store";

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
  const togglePainPoint = useOnboardingWizardStore((s) => s.togglePainPoint);
  const setOtherPainPoint = useOnboardingWizardStore(
    (s) => s.setOtherPainPoint,
  );
  const nextStep = useOnboardingWizardStore((s) => s.nextStep);

  const topIDs = getTopPickIDs(role);
  const hasSomethingElse = painPoints.includes("Something else");
  const canContinue =
    painPoints.length > 0 &&
    (!hasSomethingElse || Boolean(otherPainPoint.trim()));

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
    canContinue,
    handleLaunch,
  };
}
