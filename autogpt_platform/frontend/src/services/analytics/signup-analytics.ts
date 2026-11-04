import { MarketingConsentEvent } from "@/services/analytics/posthog-events";
import { capturePostHogEvent } from "./posthog-capture";

// Only the surface: there is no user yet, and no email may ride along. Held
// until the cookie banner is answered, like every other browser event.
export function trackSignupMarketingOptOut() {
  capturePostHogEvent(MarketingConsentEvent.MARKETING_OPTED_OUT, {
    surface: "signup",
  });
}
