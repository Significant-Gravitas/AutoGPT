// This file configures the initialization of Sentry on the client.
// The config you add here will be used whenever a users loads a page in their browser.
// https://docs.sentry.io/platforms/javascript/guides/nextjs/

import { isNextRSCNavigationFallback } from "@/lib/sentry-filters";
import { setupSessionReplay } from "@/lib/session-replay";
import { environment } from "@/services/environment";
import * as Sentry from "@sentry/nextjs";

const shouldEnable = environment.isSentryEnabled();

Sentry.init({
  dsn: "https://fe4e4aa4a283391808a5da396da20159@o4505260022104064.ingest.us.sentry.io/4507946746380288",

  environment: environment.getEnvironmentStr(),

  enabled: shouldEnable,

  // Suppress cross-origin stylesheet errors from Sentry Replay (rrweb)
  // serializing DOM snapshots with cross-origin stylesheets
  // (e.g., from browser extensions or CDN-loaded CSS)
  ignoreErrors: [
    /Not allowed to access cross-origin stylesheet/,
    // Sentry SDK internal issue on some mobile browsers
    /Error invoking postEvent: Method not found/,
  ],

  // Next's handled RSC fetch fallback reaches us only through console capture
  // and falls back to a full navigation, so it is noise (BUILDER-3QB).
  beforeSend(event) {
    return isNextRSCNavigationFallback(event) ? null : event;
  },

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

  // Set profilesSampleRate to 1.0 to profile every transaction.
  // Since profilesSampleRate is relative to tracesSampleRate,
  // the final profiling rate can be computed as tracesSampleRate * profilesSampleRate
  // For example, a tracesSampleRate of 0.5 and profilesSampleRate of 0.5 would
  // result in 25% of transactions being profiled (0.5*0.5=0.25)
  profilesSampleRate: 1.0,
  enableLogs: true,
});

export const onRouterTransitionStart = Sentry.captureRouterTransitionStart;
