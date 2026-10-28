import type { OnboardingWizardProgress } from "@/app/api/__generated__/models/onboardingWizardProgress";

export function makeProgress(
  overrides: Partial<OnboardingWizardProgress> = {},
): OnboardingWizardProgress {
  return {
    version: 1,
    currentStep: "role",
    completedSteps: [],
    role: "Engineering",
    otherRole: "",
    painPoints: ["Reporting"],
    otherPainPoint: "",
    selectedBilling: "monthly",
    hasUserSelectedBilling: false,
    selectedCountryCode: "US",
    hiredTemplateIds: [],
    ...overrides,
  };
}
