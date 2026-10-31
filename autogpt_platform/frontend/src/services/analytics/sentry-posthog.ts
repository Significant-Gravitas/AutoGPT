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
    if (isPostHogRateLimitNotice(event) || isFailedHTTPResponse(event)) {
      return event;
    }
    try {
      if (
        environment.isPostHogEnabled() &&
        hasConsentFor("analytics") &&
        !posthog.has_opted_out_capturing()
      ) {
        integration.processEvent(withMessageAsException(event));
      }
    } catch {
      // An analytics failure must not drop the original Sentry event.
    }
    return event;
  };
}

// A message Sentry records at error level, such as a console error, a failed
// API request the app reports or a WebSocket error, carries no exception, and
// PostHog's error tracking can't group an exception event with an empty list.
// The mirror gets the message as its exception; Sentry's event is unchanged
// apart from the PostHog links the integration adds to its tags. With no
// frames PostHog groups by the text, so query strings and IDs come out of it
// and one message about many URLs or blocks stays one issue.
function withMessageAsException(event: ErrorEvent): ErrorEvent {
  if (event.exception?.values?.length || !event.message) return event;
  event.tags ??= {};
  return {
    ...event,
    exception: {
      values: [
        {
          type: "Error",
          value: groupableText(event.message),
          mechanism: { type: "sentry-message", handled: true },
        },
      ],
    },
  };
}

// A query runs to the next space, double quote or backtick (an apostrophe
// can sit inside one); colons, commas, full stops and closing brackets just
// before a space end the sentence, not the query (`…/out.png?sig=1: 403`,
// `(see …?id=1)`).
const URL_QUERY = /\?[^\s"`]*?(?=[:,.;)]*(?:[\s"`]|$))/g;
const UUID =
  /\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b/gi;

// Never empty: posthog-js falls back to the raw message, query and all, when
// the exception's value is blank.
function groupableText(message: string) {
  const text = message.replace(URL_QUERY, "").replace(UUID, "{id}").trim();
  return text || "Error message with no text but a URL query";
}

// posthog-js's own wording, matched whole so an app error that merely quotes
// it still mirrors. sentry-posthog-instrumentation.test.ts drives the real
// rate limiter, so a reworded notice fails CI instead of looping silently.
function isPostHogRateLimitNotice(event: ErrorEvent) {
  return (
    event.logger === "console" &&
    event.message ===
      "[PostHog.js] This capture call is ignored due to client rate limiting."
  );
}

// A failed response the HttpClient integration reports is not a bug in the
// page, and one from the PostHog relay would be mirrored back into PostHog.
// These events carry no level today, so the allow-list skips them already;
// this keeps them out if Sentry starts setting one.
function isFailedHTTPResponse(event: ErrorEvent) {
  return Boolean(
    event.exception?.values?.some((exception) =>
      exception.mechanism?.type?.startsWith("auto.http.client."),
    ),
  );
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
