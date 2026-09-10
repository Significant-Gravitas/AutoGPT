// Hire-funnel events. The backend `hire_completed` record stays the
// source-of-truth count; these answer how long the flow took and where people
// drop out of it. Capture is best-effort — a blocked analytics host must
// never interrupt a hire.

import posthog from "posthog-js";

type ExpertsEvent =
  | "hire_started"
  | "hire_flow_completed"
  | "hire_flow_abandoned";

export function trackExperts(
  event: ExpertsEvent,
  properties?: Record<string, unknown>,
) {
  try {
    posthog.capture(event, properties);
  } catch {
    // Analytics is never worth a broken hire.
  }
}
