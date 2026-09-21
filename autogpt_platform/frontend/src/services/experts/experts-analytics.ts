// Experts + home funnel events (SECRT-2526 / SECRT-2552). Capture is
// best-effort: a blocked analytics host must never break a hire, a briefing, or
// a home render. The breadcrumb puts the same step on the timeline of any
// error Sentry records afterwards.

import * as Sentry from "@sentry/nextjs";
import posthog from "posthog-js";

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

// The ``expert_published`` wrapper lands with the PR that calls it — knip
// fails an export nothing uses yet.
type ExpertsEvent =
  | "hire_flow_completed"
  | "hire_flow_abandoned"
  | "expert_exported"
  | "expert_downloaded"
  | "expert_imported"
  | "expert_published";

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

/** How much of an expert moved: exports that dwarf imports mean loss. */
export interface ExpertPortabilityPayload {
  expert_id: string;
  workflow_count: number;
  skill_count: number;
}

export function trackExpertExported(payload: ExpertPortabilityPayload) {
  trackExperts("expert_exported", { ...payload });
}

export function trackExpertDownloaded(payload: ExpertPortabilityPayload) {
  trackExperts("expert_downloaded", { ...payload });
}

export function trackExpertImported(payload: ExpertPortabilityPayload) {
  trackExperts("expert_imported", { ...payload });
}
