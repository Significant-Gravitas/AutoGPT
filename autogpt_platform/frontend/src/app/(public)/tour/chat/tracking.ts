import { analytics } from "@/services/analytics";

export type TourCtaLabel =
  | "pricing"
  | "another-scenario"
  | "self-host"
  | "share";

const TOUR_START_SESSION_KEY = "tour_start_tracked";

export function trackTourStart() {
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
}

export function trackTourScenarioComplete(scenarioId: string) {
  analytics.sendDatafastEvent("tour_scenario_complete", {
    scenario: scenarioId,
  });
}

export function trackTourCtaClick(
  label: TourCtaLabel,
  metadata: Record<string, unknown> = {},
) {
  analytics.sendDatafastEvent("tour_cta_click", { label, ...metadata });
}
