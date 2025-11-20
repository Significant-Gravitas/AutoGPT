// This file configures the initialization of Sentry on the client.
// The config you add here will be used whenever a users loads a page in their browser.
// https://docs.sentry.io/platforms/javascript/guides/nextjs/

import { consent } from "@/services/consent/cookies";
import { environment } from "@/services/environment";
import * as Sentry from "@sentry/nextjs";

const isProdOrDev = environment.isProd() || environment.isDev();
const isCloud = environment.isCloud();
const isDisabled = process.env.DISABLE_SENTRY === "true";

const shouldEnable = !isDisabled && isProdOrDev && isCloud;

// Check for monitoring consent (includes session replay)
const hasMonitoringConsent = consent.hasConsentFor("monitoring");

Sentry.init({
  dsn: "https://fe4e4aa4a283391808a5da396da20159@o4505260022104064.ingest.us.sentry.io/4507946746380288",

  environment: environment.getEnvironmentStr(),

  enabled: shouldEnable,

  // Add optional integrations for additional features
  integrations: [
    Sentry.captureConsoleIntegration(),
    Sentry.extraErrorDataIntegration(),
    Sentry.browserProfilingIntegration(),
    Sentry.httpClientIntegration(),
    Sentry.launchDarklyIntegration(),
    Sentry.replayIntegration({
      unmask: [".sentry-unmask, [data-sentry-unmask]"],
    }),
    Sentry.replayCanvasIntegration(),
    Sentry.reportingObserverIntegration(),
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
  // GDPR: Only enable if user has consented to monitoring
  replaysSessionSampleRate: hasMonitoringConsent ? 0.1 : 0,

  // Define how likely Replay events are sampled when an error occurs.
  // GDPR: Only enable if user has consented to monitoring
  replaysOnErrorSampleRate: hasMonitoringConsent ? 1.0 : 0,

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
