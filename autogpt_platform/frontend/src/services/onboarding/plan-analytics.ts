// The onboarding plan's events: the plan step being seen, and its team being
// hired once the trial is confirmed. Best-effort, like every funnel event: a
// blocked analytics host must never hold up the step or the hire.

import { PlanEvent } from "@/services/analytics/posthog-events";
import posthog from "posthog-js";

interface PlanEventProperties {
  [PlanEvent.PLAN_VIEWED]: {
    source: "llm" | "fallback";
    skipped_dump: boolean;
    tasks: number;
    experts: number;
    wait_ms: number;
    gave_up: boolean;
  };
  [PlanEvent.PLAN_EXPERTS_HIRED]: {
    hired: number;
    failed: number;
    via: "preparing";
  };
}

export function trackPlan<Event extends keyof PlanEventProperties>(
  event: Event,
  properties: PlanEventProperties[Event],
) {
  try {
    posthog.capture(event, properties);
  } catch {
    // Analytics is never worth a broken onboarding step.
  }
}
