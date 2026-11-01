// Composer dictation recovery. Capture is best-effort — a blocked analytics
// host must never interrupt someone mid-recording.

import {
  DictationEvent,
  type EventName,
} from "@/services/analytics/posthog-events";
import posthog from "posthog-js";

type DictationEventName = EventName<typeof DictationEvent>;

export function trackDictation(event: DictationEventName) {
  try {
    posthog.capture(event);
  } catch {
    // Analytics is never worth a broken recording.
  }
}
