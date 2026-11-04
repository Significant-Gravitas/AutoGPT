// This file configures the initialization of Sentry on the client.
// The config you add here will be used whenever a users loads a page in their browser.
// https://docs.sentry.io/platforms/javascript/guides/nextjs/

import { isLoggedRequestFailure } from "@/app/api/mutators/request-failure-log";
import { reloadOnUnhandledChunkLoadError } from "@/lib/chunk-load-reload";
import {
  CLIENT_IGNORE_ERRORS,
  isFromBot,
  isInjectedIframeProbe,
  isNextRSCNavigationFallback,
  SENTRY_DATA_COLLECTION,
  stripLogUserPII,
  stripSpanDataPII,
  stripSpanPII,
  stripUserPII,
  tagDeploySkew,
} from "@/lib/sentry-filters";
import { getSentryTunnelURL } from "@/lib/sentry-tunnel";
import { redactEventURLs, redactSpanURLs } from "@/lib/sentry-url-redaction";
import { createPostHogSentryProcessor } from "@/services/analytics/sentry-posthog";
import { environment } from "@/services/environment";
import * as Sentry from "@sentry/nextjs";

const dsn = environment.getSentryDSN();
const mirrorSentryEvent = createPostHogSentryProcessor(dsn);

Sentry.init({
  dsn,

  environment: environment.getSentryEnvironment(),

  enabled: Boolean(dsn),

  // The tunnel is same-origin, so fetch would attach our cookies to every envelope.
  tunnel: getSentryTunnelURL(dsn),
  transportOptions: { fetchOptions: { credentials: "omit" } },

  ignoreErrors: CLIENT_IGNORE_ERRORS,

  // Errors are reported without consent (legitimate interest), so they carry
  // the user id only: no email, and Sentry is told not to infer the IP address.
  dataCollection: SENTRY_DATA_COLLECTION,

  // Only errors carry a stack trace, so a console warning or captured message
  // neither opens a new issue group nor marks the session as errored.
  attachStacktrace: false,

  // Send spans inside their transaction, which beforeSendTransaction and the
  // span filters below are written for; streamed spans would skip them.
  traceLifecycle: "static",

  // Next's handled RSC fetch fallback reaches us only through console capture
  // and falls back to a full navigation, so it is noise (BUILDER-3QB).
  // Crawlers render our pages too; what breaks for them is not a user's bug.
  // A failed API request the request logger reports, titled by its route,
  // would otherwise arrive again from the HttpClient integration (BUILDER-95D).
  // Some browsers inject a script that logs every cross-origin iframe it may
  // not read (BUILDER-943). URLs are masked last: the request logger's match
  // needs the URL as it was requested.
  beforeSend(event) {
    stripSpanPII(stripUserPII(event));
    if (isFromBot(navigator.userAgent)) return null;
    if (isLoggedRequestFailure(event)) return null;
    if (isInjectedIframeProbe(event)) return null;
    if (isNextRSCNavigationFallback(event)) return null;
    // Mirror last: after our filters (an integration would run before
    // beforeSend) and on the stripped, URL-masked event, so PostHog never
    // gets an email, IP address or credential Sentry doesn't keep either.
    return mirrorSentryEvent(redactEventURLs(tagDeploySkew(event)));
  },
  beforeSendTransaction(event) {
    return isFromBot(navigator.userAgent)
      ? null
      : stripSpanPII(stripUserPII(redactEventURLs(event)));
  },
  beforeSendSpan: Sentry.withStaticSpan((span) =>
    redactSpanURLs(stripSpanDataPII(span)),
  ),
  beforeSendLog: stripLogUserPII,

  // Add optional integrations for additional features
  integrations: [
    // A new session on every navigation, not only on page load.
    Sentry.browserSessionIntegration({ lifecycle: "route" }),
    // Spans for performance.mark() and performance.measure().
    Sentry.userTimingIntegration(),
    Sentry.captureConsoleIntegration({ levels: ["fatal", "error", "warn"] }),
    Sentry.extraErrorDataIntegration(),
    Sentry.browserProfilingIntegration(),
    Sentry.httpClientIntegration(),
    Sentry.featureFlagsIntegration(),
    // Deprecation reports are browser platform notices about the web platform
    // itself (e.g. Chrome's "Attribution Reporting is deprecated"), not bugs in
    // our code, and they bury real issues. Crash and intervention reports still
    // come through.
    Sentry.reportingObserverIntegration({ types: ["crash", "intervention"] }),
    // Sentry.feedbackIntegration({
    //   // Additional SDK configuration goes in here, for example:
    //   colorScheme: "system",
    // }),
  ],

  // Define how likely traces are sampled. Adjust this value in production, or use tracesSampler for greater control.
  tracesSampleRate: 1,

  // Set `tracePropagationTargets` to control for which URLs trace propagation should be enabled
  tracePropagationTargets: [
    "localhost",
    "localhost:8006",
    /^https:\/\/dev\-builder\.agpt\.co\/api/,
    /^https:\/\/.*\.agpt\.co\/api/,
  ],

  // Setting this option to true will print useful information to the console while you're setting up Sentry.
  debug: false,

  // Profiling samples the main thread every 10 ms while a trace is running, so
  // profiling every session slowed every session down. 1 session in 10 is
  // profiled. Traces stay at 100%: the backend's traces_sampler follows the
  // browser's decision for the requests it receives.
  profileSessionSampleRate: 0.1,
  profileLifecycle: "trace",
});

reloadOnUnhandledChunkLoadError();

export function onRouterTransitionStart(
  ...args: Parameters<typeof Sentry.captureRouterTransitionStart>
) {
  Sentry.captureRouterTransitionStart(...args);
}
