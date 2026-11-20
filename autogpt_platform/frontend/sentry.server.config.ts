// This file configures the initialization of Sentry on the server.
// The config you add here will be used whenever the server handles a request.
// https://docs.sentry.io/platforms/javascript/guides/nextjs/

import {
  SENTRY_DATA_COLLECTION,
  stripLogUserPII,
  stripRequestCredentials,
  stripSpanDataPII,
  stripSpanPII,
  stripUserPII,
} from "@/lib/sentry-filters";
import { isSentryTunnelTransaction } from "@/lib/sentry-tunnel";
import { redactEventURLs, redactSpanURLs } from "@/lib/sentry-url-redaction";
import { environment } from "@/services/environment";
import * as Sentry from "@sentry/nextjs";
// import { NodeProfilingIntegration } from "@sentry/profiling-node";

const dsn = environment.getSentryDSN();

Sentry.init({
  dsn,

  environment: environment.getSentryEnvironment(),

  enabled: Boolean(dsn),

  // Define how likely traces are sampled. Adjust this value in production, or use tracesSampler for greater control.
  tracesSampleRate: 1,
  tracePropagationTargets: [
    "localhost",
    "localhost:8006",
    /^https:\/\/dev\-builder\.agpt\.co\/api/,
    /^https:\/\/.*\.agpt\.co\/api/,
  ],

  // No email or IP address reaches Sentry; the user id is enough.
  dataCollection: SENTRY_DATA_COLLECTION,

  // Only errors carry a stack trace, so a captured message doesn't open a new
  // issue group.
  attachStacktrace: false,

  // Send spans inside their transaction, which beforeSendTransaction and the
  // span filters below are written for; streamed spans would skip them.
  traceLifecycle: "static",

  beforeSend(event) {
    return stripSpanPII(
      stripUserPII(redactEventURLs(stripRequestCredentials(event))),
    );
  },
  beforeSendTransaction(event) {
    return isSentryTunnelTransaction(event)
      ? null
      : stripSpanPII(
          stripUserPII(redactEventURLs(stripRequestCredentials(event))),
        );
  },
  beforeSendSpan: Sentry.withStaticSpan((span) =>
    redactSpanURLs(stripSpanDataPII(span)),
  ),
  beforeSendLog: stripLogUserPII,

  // Setting this option to true will print useful information to the console while you're setting up Sentry.
  debug: false,

  // Integrations
  integrations: [
    Sentry.requestDataIntegration({ include: { cookies: false, data: false } }),
    // Vercel freezes an idle function between requests, and the ANR watchdog
    // reports every thaw as a blocked event loop with no stack (BUILDER-9).
    // Only a long-lived self-hosted Node server can really hang.
    ...(environment.isVercel() ? [] : [Sentry.anrIntegration()]),
    // NodeProfilingIntegration,
    Sentry.extraErrorDataIntegration(),
    // Sentry.fsIntegration(),
  ],
});
