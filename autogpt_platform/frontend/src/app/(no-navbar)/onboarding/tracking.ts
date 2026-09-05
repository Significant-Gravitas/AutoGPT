import { analytics } from "@/services/analytics";
import type { Step } from "./store";

export type OnboardingStepKey =
  | "welcome"
  | "role"
  | "pain_points"
  | "preparing";

const SENT_KEY_PREFIX = "onboarding_step_sent_";

interface StepLayout {
  welcome: number;
  role: number;
  painPoints: number;
  preparing: number;
}

/**
 * Maps a wizard step number to a stable key.
 *
 * Step numbers differ between the paywall-first and no-paywall layouts, so the
 * number alone is meaningless across cohorts — Welcome is 2 for one and 1 for
 * the other. Keying by name keeps a single funnel readable for both. The
 * subscription step returns null: it is reported as `paywall_view` instead.
 */
export function onboardingStepKey(
  steps: StepLayout,
  step: Step,
): OnboardingStepKey | null {
  if (step === steps.welcome) return "welcome";
  if (step === steps.role) return "role";
  if (step === steps.painPoints) return "pain_points";
  if (step === steps.preparing) return "preparing";
  return null;
}

/**
 * Reports that the user reached a wizard step, at most once per tab.
 *
 * The wizard's five steps share the `/onboarding` URL and the backend records
 * only ONBOARDING_COMPLETE, so without this nothing distinguishes "abandoned on
 * Role" from "abandoned on Preparing" — the drop between signup and copilot is
 * a single opaque number.
 *
 * One goal per step rather than one goal carrying the step as metadata, because
 * DataFast funnel steps match on goal name; metadata could not express an
 * ordered funnel.
 */
export function trackOnboardingStep(key: OnboardingStepKey) {
  const sentKey = `${SENT_KEY_PREFIX}${key}`;
  try {
    if (sessionStorage.getItem(sentKey)) return;
    sessionStorage.setItem(sentKey, "1");
  } catch {
    // In-app browsers may block sessionStorage — double-counting a step beats
    // dropping it, so fall through to report either way.
  }

  try {
    analytics.sendDatafastEvent(`onboarding_${key}`, {});
  } catch {
    // sendDatafastEvent hands off to the third-party DataFast script, which it
    // calls without a guard of its own. This runs inside the onboarding wizard,
    // so an exception here would unmount the flow into an error boundary.
  }
}
