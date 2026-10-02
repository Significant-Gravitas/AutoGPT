import { SignupEvent } from "@/services/analytics/posthog-events";
import posthog from "posthog-js";

// Sent without properties: there is no user yet, and no email may ride along.
export function trackSignupMarketingOptOut() {
  try {
    posthog.capture(SignupEvent.SIGNUP_MARKETING_OPT_OUT);
  } catch {
    // A blocked analytics host must never break signup.
  }
}
