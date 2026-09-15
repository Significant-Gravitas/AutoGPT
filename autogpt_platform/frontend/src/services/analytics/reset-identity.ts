import posthog from "posthog-js";
import { resetAnonymousID } from "./anonymous-id";

export function resetAnalyticsIdentity(): void {
  let anonymousID: string | undefined;
  try {
    if (posthog.get_distinct_id()) {
      posthog.reset(true);
      anonymousID = posthog.get_distinct_id();
    }
  } catch {
    anonymousID = undefined;
  }
  resetAnonymousID(anonymousID);
}
