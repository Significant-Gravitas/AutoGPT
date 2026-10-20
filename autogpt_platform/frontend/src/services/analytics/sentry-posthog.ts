import { hasConsentFor } from "@/services/consent/consent";
import { environment, HOSTED_SENTRY_DSN } from "@/services/environment";
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

// The organization the hosted DSN reports to; next.config.mjs uploads source
// maps under the same slug.
const SENTRY_ORG_SLUG = "significant-gravitas";
const SENTRY_INGEST_HOST = /^(o\d+)\.ingest(?:\.[a-z]{2})?\.sentry\.io$/;

function getIngestOrg(dsn: string) {
  return new URL(dsn).hostname.match(SENTRY_INGEST_HOST)?.[1];
}

function getSentryProject(dsn: string | undefined) {
  if (!dsn) return {};
  try {
    const projectID = Number(new URL(dsn).pathname.slice(1));
    const org = getIngestOrg(dsn);
    if (
      org &&
      org === getIngestOrg(HOSTED_SENTRY_DSN) &&
      Number.isSafeInteger(projectID) &&
      projectID > 0
    ) {
      return { organization: SENTRY_ORG_SLUG, projectId: projectID };
    }
  } catch {
    // Unknown DSNs can still receive PostHog links without a guessed Sentry URL.
  }
  return {};
}
