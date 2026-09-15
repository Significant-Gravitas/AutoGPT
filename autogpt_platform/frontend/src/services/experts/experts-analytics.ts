// Hire-funnel events. The backend `hire_completed` record stays the
// source-of-truth count; these answer how long the flow took and where people
// drop out of it. Capture is best-effort — a blocked analytics host must
// never interrupt a hire.

import posthog from "posthog-js";

// The ``expert_published`` wrapper lands with the PR that calls it — knip
// fails an export nothing uses yet.
type ExpertsEvent =
  | "hire_started"
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
