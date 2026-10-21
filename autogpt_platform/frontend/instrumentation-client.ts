// This file configures the initialization of Sentry on the client.
// The config you add here will be used whenever a users loads a page in their browser.
// https://docs.sentry.io/platforms/javascript/guides/nextjs/

import { isLoggedRequestFailure } from "@/app/api/mutators/request-failure-log";
import {
  CLIENT_IGNORE_ERRORS,
  isFromBot,
  isInjectedIframeProbe,
  isNextRSCNavigationFallback,
  stripLogUserPII,
  stripSpanDataPII,
  stripSpanPII,
  stripUserPII,
} from "@/lib/sentry-filters";
import { getSentryTunnelURL } from "@/lib/sentry-tunnel";
import { redactEventURLs, redactSpanURLs } from "@/lib/sentry-url-redaction";
import { holdReplayBackFor, setupSessionReplay } from "@/lib/session-replay";
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

  // Suppress cross-origin stylesheet errors from Sentry Replay (rrweb)
  // serializing DOM snapshots with cross-origin stylesheets
  // (e.g., from browser extensions or CDN-loaded CSS)
  ignoreErrors: CLIENT_IGNORE_ERRORS,

  // Errors are reported without consent (legitimate interest), so they carry
  // the user id only: no email, and Sentry is told not to infer the IP address.
  sendDefaultPii: false,

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
    return mirrorSentryEvent(redactEventURLs(event));
  },
  beforeSendTransaction(event) {
    return isFromBot(navigator.userAgent)
      ? null
      : stripSpanPII(stripUserPII(redactEventURLs(event)));
  },
  beforeSendSpan: (span) => redactSpanURLs(stripSpanDataPII(span)),
  beforeSendLog: stripLogUserPII,

  // Add optional integrations for additional features
  integrations: [
    Sentry.captureConsoleIntegration({ levels: ["fatal", "error", "warn"] }),
    Sentry.extraErrorDataIntegration(),
    Sentry.browserProfilingIntegration(),
    Sentry.httpClientIntegration(),
    Sentry.featureFlagsIntegration(),
    // GDPR: session replay only once the visitor consents to monitoring
    ...setupSessionReplay(),
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

  // Define how likely Replay events are sampled.
  // This sets the sample rate to be 10%. You may want this to be 100% while
  // in development and sample at a lower rate in production
  // Inert until setupSessionReplay installs the replay integration.
  replaysSessionSampleRate: 0.1,

  // Define how likely Replay events are sampled when an error occurs.
  replaysOnErrorSampleRate: 1.0,

  // Setting this option to true will print useful information to the console while you're setting up Sentry.
  debug: false,

  // Profiling samples the main thread every 10 ms for as long as a transaction
  // lasts (up to 30 s a page load), so profiling every transaction slowed every
  // session down. It is relative to tracesSampleRate, so with every transaction
  // traced, 1 in 10 is profiled. Traces stay at 100%: the backend's
  // traces_sampler follows the browser's decision for the requests it receives.
  profilesSampleRate: 0.1,
  enableLogs: true,
});

export function onRouterTransitionStart(
  ...args: Parameters<typeof Sentry.captureRouterTransitionStart>
) {
  Sentry.captureRouterTransitionStart(...args);
  holdReplayBackFor(args[0]);
}
