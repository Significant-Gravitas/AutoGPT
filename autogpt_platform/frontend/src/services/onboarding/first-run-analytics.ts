// First-run walkthrough and contextual-tip funnel (SECRT-3149). Separate from
// brain-dump-analytics: the walkthrough is replayable from the profile menu
// long after the dump funnel is over.

import {
  FirstRunEvent,
  type EventName,
} from "@/services/analytics/posthog-events";
import posthog from "posthog-js";

type FirstRunEventName = EventName<typeof FirstRunEvent>;

// `source` is the emitter's own super property (`web` here); an event
// property of that name would overwrite it.
type FirstRunProperties = Record<string, string | number | boolean | null> & {
  source?: never;
};

export function trackFirstRun(
  event: FirstRunEventName,
  properties: FirstRunProperties,
) {
  try {
    posthog.capture(event, properties);
  } catch {
    // A blocked analytics host must never break a first visit.
  }
}
