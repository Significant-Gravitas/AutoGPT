// Pages: whether people open the pages Otto composes, and run from them.
// Capture is best-effort; a blocked analytics host must never stop a run.

import { PagesEvent } from "@/services/analytics/posthog-events";
import posthog from "posthog-js";

export function trackPageOpened(surface: "pages" | "library") {
  capture(PagesEvent.PAGE_OPENED, { surface });
}

export function trackPageRunStarted(
  source: "form" | "quick_run" | "row_button",
) {
  capture(PagesEvent.PAGE_RUN_STARTED, { source });
}

function capture(event: string, properties: Record<string, unknown>) {
  try {
    posthog.capture(event, properties);
  } catch {
    // Analytics is never worth a failed run.
  }
}
