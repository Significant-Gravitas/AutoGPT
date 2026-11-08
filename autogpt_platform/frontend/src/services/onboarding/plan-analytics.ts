// The onboarding plan's events: the plan step being seen. Its team being hired
// is an outcome, sent by the backend. Best-effort, like every funnel event: a
// blocked analytics host must never hold up the step.

import { PlanEvent } from "@/services/analytics/posthog-events";
import posthog from "posthog-js";

interface PlanEventProperties {
  [PlanEvent.PLAN_VIEWED]: {
    // Not `source`: that base property names the emitter (`web`).
    plan_source: "llm" | "fallback";
    skipped_dump: boolean;
    tasks: number;
    experts: number;
    wait_ms: number;
    gave_up: boolean;
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
