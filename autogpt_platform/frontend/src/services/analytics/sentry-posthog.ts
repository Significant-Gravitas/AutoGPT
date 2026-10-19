import { hasConsentFor } from "@/services/consent/consent";
import { environment } from "@/services/environment";
import type { ErrorEvent } from "@sentry/nextjs";
import posthog from "posthog-js";

export function createPostHogSentryProcessor(dsn: string | undefined) {
  const integration = posthog.sentryIntegration({
    ...getSentryProject(dsn),
    severityAllowList: ["error", "fatal"],
  });

  return function mirrorSentryEvent(event: ErrorEvent) {
    if (
      event.logger === "console" &&
      event.message ===
        "[PostHog.js] This capture call is ignored due to client rate limiting."
    ) {
      return event;
    }
    try {
      if (
        environment.isPostHogEnabled() &&
        hasConsentFor("analytics") &&
        !posthog.has_opted_out_capturing()
      ) {
        integration.processEvent(event);
      }
    } catch {
      // An analytics failure must not drop the original Sentry event.
    }
    return event;
  };
}

function getSentryProject(dsn: string | undefined) {
  if (!dsn) return {};
  try {
    const { hostname, pathname } = new URL(dsn);
    const projectID = Number(pathname.slice(1));
    if (
      /^o4505260022104064\.ingest(?:\.[a-z]{2})?\.sentry\.io$/.test(hostname) &&
      Number.isSafeInteger(projectID) &&
      projectID > 0
    ) {
      return { organization: "significant-gravitas", projectId: projectID };
    }
  } catch {
    // Unknown DSNs can still receive PostHog links without a guessed Sentry URL.
  }
  return {};
}
