import { capturePostHogEvent } from "@/services/analytics/posthog-capture";
import { OnboardingEvent } from "@/services/analytics/posthog-events";
import type { StepLayout } from "./store";

export type OnboardingStepKey =
  | "team"
  | "autopilot"
  | "role"
  | "pain_points"
  | "connect"
  | "hire"
  | "preparing";

const POSTHOG_SENT_KEY_PREFIX = "posthog_onboarding_step_viewed_";

/**
 * Maps a wizard step number to a stable key.
 *
 * Step numbers differ between layouts, so the number alone is meaningless
 * across cohorts — Preparing is 4 without a paywall and 5 with one. Keying by
 * name keeps a single funnel readable for all. The subscription step returns
 * null: it is reported as `paywall_viewed` instead.
 */
export function onboardingStepKey(
  steps: StepLayout,
  step: number,
): OnboardingStepKey | null {
  if (step === steps.team) return "team";
  if (step === steps.autopilot) return "autopilot";
  if (step === steps.role) return "role";
  if (step === steps.painPoints) return "pain_points";
  if (step === steps.connect) return "connect";
  if (step === steps.hire) return "hire";
  if (step === steps.preparing) return "preparing";
  return null;
}

/**
 * Reports that the user reached a wizard step, at most once per tab.
 *
 * The wizard's steps share the `/onboarding` URL and the backend records
 * only ONBOARDING_COMPLETE, so without this nothing distinguishes "abandoned on
 * Role" from "abandoned on Preparing" — the drop between signup and copilot is
 * a single opaque number. PostHog funnels order steps by their filter, so one
 * event with a `step` property serves every step.
 *
 * The once-per-tab guard is marked only once the event is sent or dropped for
 * lack of consent, not while it waits for the answer.
 */
export function trackOnboardingStep(key: OnboardingStepKey) {
  capturePostHogEvent(
    OnboardingEvent.ONBOARDING_STEP_VIEWED,
    { step: key },
    { oncePerTabKey: `${POSTHOG_SENT_KEY_PREFIX}${key}` },
  );
}
