// Experts + home funnel events (SECRT-2526 / SECRT-2552). Capture is
// best-effort: a blocked analytics host must never break a hire, a briefing, or
// a home render. The breadcrumb puts the same step on the timeline of any
// error Sentry records afterwards.

import {
  ExpertsFunnelEvent,
  HireFlowEvent,
  type EventName,
} from "@/services/analytics/posthog-events";
import type { HireRequestSurface } from "@/app/api/__generated__/models/hireRequestSurface";
import * as Sentry from "@sentry/nextjs";
import posthog from "posthog-js";

export type FunnelViewEvent =
  | typeof ExpertsFunnelEvent.EXPERTS_SECTION_VIEWED
  | typeof ExpertsFunnelEvent.HOME_VIEWED
  | typeof ExpertsFunnelEvent.BRIEFING_OPENED;

interface FunnelEventProperties {
  [ExpertsFunnelEvent.EXPERT_PROFILE_OPENED]: { template_id: string };
  [ExpertsFunnelEvent.HIRE_STARTED]: {
    template_id: string;
    surface: NonNullable<HireRequestSurface>;
  };
  [ExpertsFunnelEvent.EXPERT_THREAD_CREATED]: { expert_id: string };
  [ExpertsFunnelEvent.BRIEFING_OUTCOME_CLICKED]: { status: string };
  [ExpertsFunnelEvent.HOME_ATTENTION_ACTIONED]: {
    kind: string;
    action: "approve" | "decline";
  };
  [ExpertsFunnelEvent.HOME_TEAM_MEMBER_CLICKED]: { expert_id: string };
}

type FunnelEvent = FunnelViewEvent | keyof FunnelEventProperties;

export function trackFunnel(event: FunnelViewEvent): void;
export function trackFunnel<Event extends keyof FunnelEventProperties>(
  event: Event,
  properties: FunnelEventProperties[Event],
): void;

export function trackFunnel(
  event: FunnelEvent,
  properties?: Record<string, unknown>,
) {
  // Separate boundaries: a breadcrumb failure must not cost the capture.
  try {
    Sentry.addBreadcrumb({
      category: "funnel",
      message: event,
      data: properties ?? {},
      level: "info",
    });
  } catch {
    // Analytics is never worth a broken interaction.
  }

  try {
    posthog.capture(event, properties);
  } catch {
    // Analytics is never worth a broken interaction.
  }
}

type ExpertsEvent = EventName<typeof HireFlowEvent>;

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
