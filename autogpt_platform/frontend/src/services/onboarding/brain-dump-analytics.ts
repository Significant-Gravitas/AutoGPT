// Brain-dump funnel events. These are the release-blocker instrumentation:
// completion rate, dump length distribution, and every way a dump can go
// wrong. Capture is best-effort — a blocked analytics host must never
// interrupt someone mid-recording.
//
// Lives in services/ rather than beside the onboarding step because the
// copilot home fires the tail of this funnel (the intro and the follow-up
// dump), and a feature folder reaching into another feature's internals
// is how import graphs rot.

import {
  BrainDumpEvent,
  type EventName,
} from "@/services/analytics/posthog-events";
import posthog from "posthog-js";

type BrainDumpEventName = EventName<typeof BrainDumpEvent>;

export function trackBrainDump(
  event: BrainDumpEventName,
  properties?: Record<string, unknown>,
) {
  try {
    posthog.capture(event, properties);
  } catch {
    // Analytics is never worth a broken recording.
  }
}
