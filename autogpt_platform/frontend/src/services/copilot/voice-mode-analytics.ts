// Voice-mode funnel. The question this has to answer is whether anyone uses
// it twice: turns per session, and every way the loop can end badly.
// Capture is best-effort — a blocked analytics host must never interrupt
// someone mid-sentence.
//
// Lives in services/ alongside the brain-dump funnel because both measure
// the same microphone path from different ends.

import {
  VoiceModeEvent,
  type EventName,
} from "@/services/analytics/posthog-events";
import posthog from "posthog-js";

type VoiceModeEventName = EventName<typeof VoiceModeEvent>;

export function trackVoiceMode(
  event: VoiceModeEventName,
  properties?: Record<string, unknown>,
) {
  try {
    posthog.capture(event, properties);
  } catch {
    // Analytics is never worth a broken conversation.
  }
}
