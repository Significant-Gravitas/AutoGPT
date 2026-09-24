import { analytics } from "@/services/analytics";
import { capturePostHogEvent } from "@/services/analytics/posthog-capture";
import { TourEvent } from "@/services/analytics/posthog-events";

// Each tour step goes to DataFast and, under its PostHog name, to PostHog.
// The tour is public and pre-signup: nothing here may carry an identifier.

export type TourCtaLabel =
  | "pricing"
  | "another-scenario"
  | "self-host"
  | "share";

const TOUR_START_SESSION_KEY = "tour_start_tracked";
const POSTHOG_TOUR_START_KEY = "posthog_tour_started";

export function trackTourStart() {
  // PostHog's guard is marked only once the event is sent or dropped for lack
  // of consent, not while it waits for the answer.
  capturePostHogEvent(
    TourEvent.TOUR_STARTED,
    {},
    { oncePerTabKey: POSTHOG_TOUR_START_KEY },
  );
  try {
    if (sessionStorage.getItem(TOUR_START_SESSION_KEY)) return;
    sessionStorage.setItem(TOUR_START_SESSION_KEY, "1");
  } catch {
    // In-app browsers may block sessionStorage — double-counting a visit
    // beats dropping it.
  }
  analytics.sendDatafastEvent("tour_start", {});
}

export function trackTourScenarioStart(scenarioId: string) {
  analytics.sendDatafastEvent("tour_scenario_start", { scenario: scenarioId });
  capturePostHogEvent(TourEvent.TOUR_SCENARIO_STARTED, {
    scenario: scenarioId,
  });
}

export function trackTourScenarioComplete(scenarioId: string) {
  analytics.sendDatafastEvent("tour_scenario_complete", {
    scenario: scenarioId,
  });
  capturePostHogEvent(TourEvent.TOUR_SCENARIO_COMPLETED, {
    scenario: scenarioId,
  });
}

export function trackTourCtaClick(
  label: TourCtaLabel,
  metadata: Record<string, unknown> = {},
) {
  analytics.sendDatafastEvent("tour_cta_click", { label, ...metadata });
  capturePostHogEvent(TourEvent.TOUR_CTA_CLICKED, { label, ...metadata });
}
