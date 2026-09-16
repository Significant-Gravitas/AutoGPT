// Experts + home funnel events (SECRT-2526 / SECRT-2552). Capture is
// best-effort: a blocked analytics host must never break a hire, a briefing, or
// a home render. `trackFunnel` rides the backend analytics sink
// (log_raw_analytics) so every funnel step lands in one pipeline; `trackExperts`
// stays on PostHog, where the hire-flow timing series already lives.

import posthog from "posthog-js";
import { postAnalyticsLogRawAnalytics } from "@/app/api/__generated__/endpoints/analytics/analytics";

export type FunnelViewEvent =
  | "experts_section_viewed"
  | "home_viewed"
  | "briefing_opened";

interface FunnelEventProperties {
  expert_profile_opened: { template_id: string };
  hire_started: { template_id: string };
  expert_thread_created: { expert_id: string };
  briefing_outcome_clicked: { status: string };
  home_attention_actioned: {
    kind: string;
    action: "approve" | "decline";
  };
  home_team_member_clicked: { expert_id: string };
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
  void postAnalyticsLogRawAnalytics({
    type: event,
    data: properties ?? {},
    data_index: event,
  }).catch(() => {
    // Analytics is never worth a broken interaction.
  });
}

type ExpertsEvent = "hire_flow_completed" | "hire_flow_abandoned";

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
