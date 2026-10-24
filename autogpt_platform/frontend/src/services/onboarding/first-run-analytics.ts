// First-run walkthrough and contextual-tip funnel (SECRT-3149). Separate from
// brain-dump-analytics: the walkthrough is replayable from the profile menu
// long after the dump funnel is over.

import {
  FirstRunEvent,
  type EventName,
} from "@/services/analytics/posthog-events";
import posthog from "posthog-js";

type FirstRunEventName = EventName<typeof FirstRunEvent>;

export function trackFirstRun(
  event: FirstRunEventName,
  properties: Record<string, string | number | boolean | null>,
) {
  try {
    posthog.capture(event, properties);
  } catch {
    // A blocked analytics host must never break a first visit.
  }
}
