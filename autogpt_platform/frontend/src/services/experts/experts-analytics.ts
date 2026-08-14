// Experts + home-briefing funnel events (SECRT-2526 / SECRT-2552). Capture is
// best-effort: a blocked analytics host must never break a hire, a briefing, or
// a home render. Mirrors services/onboarding/brain-dump-analytics.ts but rides
// the backend analytics sink (log_raw_analytics) instead of PostHog, so every
// funnel event lands in one pipeline.

import { postAnalyticsLogRawAnalytics } from "@/app/api/__generated__/endpoints/analytics/analytics";

type EmptyFunnelEvent =
  | "experts_section_viewed"
  | "hire_started"
  | "expert_thread_created"
  | "home_viewed"
  | "briefing_opened";

interface FunnelEventProperties {
  expert_profile_opened: { template_id: string };
  briefing_outcome_clicked: { status: string };
  home_attention_actioned: {
    kind: string;
    action: "approve" | "decline";
  };
  home_team_member_clicked: { expert_id: string };
}

type FunnelEvent = EmptyFunnelEvent | keyof FunnelEventProperties;

export function trackFunnel(event: EmptyFunnelEvent): void;
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
