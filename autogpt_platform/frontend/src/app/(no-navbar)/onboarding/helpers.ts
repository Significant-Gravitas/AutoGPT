// Shared resolution of the wizard's "Other" / "Something else" sentinels into
// real values. Both `useSubscriptionStep` (pre-Stripe-redirect) and
// `useOnboardingPage` (Preparing-step submit) post the profile, and label or
// mapping changes must stay in sync across the two.

import type { NO_PAYWALL_STEPS, PAYWALL_FIRST_STEPS, Step } from "./store";

type StepLayout = typeof PAYWALL_FIRST_STEPS | typeof NO_PAYWALL_STEPS;

export function getOnboardingStepName(args: {
  step: Step;
  steps: StepLayout;
  isBrainDumpEnabled: boolean;
}): string {
  const { step, steps, isBrainDumpEnabled } = args;
  if ("subscription" in steps && step === steps.subscription) {
    return "subscription";
  }
  if (step === steps.welcome) return "welcome";
  if (step === steps.role) return "role";
  if (step === steps.painPoints) {
    return isBrainDumpEnabled ? "brain_dump" : "pain_points";
  }
  return "preparing";
}

interface ProfileSource {
  name: string;
  role: string;
  otherRole: string;
  painPoints: string[];
  otherPainPoint: string;
}

interface NormalizedProfile {
  name: string;
  role: string;
  painPoints: string[];
}

// "brain_dump" → "Brain Dump", for the event NAME ("Onboarding Brain
// Dump Viewed") — the snake_case form stays on the `step` property.
export function onboardingStepDisplayName(stepName: string): string {
  return stepName
    .split("_")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}

// What the user actually chose on the step they just left, attached to
// onboarding_step_completed so funnels can break down by role/pain
// points/billing without per-click events. The name itself is PII the
// profile submit already owns — analytics only needs whether one was
// given. The brain-dump step's detail lives in its own brain_dump_*
// events.
export function getStepCompletionProps(
  stepName: string,
  state: ProfileSource & { selectedBilling: "monthly" | "yearly" },
): Record<string, unknown> {
  const profile = normalizeOnboardingProfile(state);
  switch (stepName) {
    case "welcome":
      return { has_name: Boolean(profile.name.trim()) };
    case "role":
      return { role: profile.role };
    case "subscription":
      return { billing: state.selectedBilling };
    default:
      return {};
  }
}

export function normalizeOnboardingProfile(
  state: ProfileSource,
): NormalizedProfile {
  const resolvedRole = state.role === "Other" ? state.otherRole : state.role;
  const resolvedPainPoints = state.painPoints
    .filter((p) => p !== "Something else")
    .concat(
      state.painPoints.includes("Something else") && state.otherPainPoint.trim()
        ? [state.otherPainPoint.trim()]
        : [],
    );
  return {
    name: state.name,
    role: resolvedRole,
    painPoints: resolvedPainPoints,
  };
}
