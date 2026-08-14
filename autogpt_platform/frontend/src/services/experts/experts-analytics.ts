// Experts + home-briefing funnel events (SECRT-2526 / SECRT-2552). Capture is
// best-effort: a blocked analytics host must never break a hire, a briefing, or
// a home render. Mirrors services/onboarding/brain-dump-analytics.ts but rides
// the backend analytics sink (log_raw_analytics) instead of PostHog, so every
// funnel event lands in one pipeline.

import { postAnalyticsLogRawAnalytics } from "@/app/api/__generated__/endpoints/analytics/analytics";

type FunnelEvent =
  | "experts_section_viewed"
  | "expert_profile_opened"
  | "hire_started"
  | "expert_thread_created"
  | "home_viewed"
  | "briefing_opened"
  | "briefing_outcome_clicked"
  | "home_attention_actioned"
  | "home_team_member_clicked";

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
