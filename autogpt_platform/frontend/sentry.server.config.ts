// This file configures the initialization of Sentry on the server.
// The config you add here will be used whenever the server handles a request.
// https://docs.sentry.io/platforms/javascript/guides/nextjs/

import { BehaveAs, getBehaveAs, getEnvironmentStr } from "@/lib/utils";
import * as Sentry from "@sentry/nextjs";
// import { NodeProfilingIntegration } from "@sentry/profiling-node";

const isProductionCloud =
  process.env.NODE_ENV === "production" && getBehaveAs() === BehaveAs.CLOUD;

Sentry.init({
  dsn: "https://fe4e4aa4a283391808a5da396da20159@o4505260022104064.ingest.us.sentry.io/4507946746380288",

  environment: getEnvironmentStr(),

  enabled: isProductionCloud,

  // Define how likely traces are sampled. Adjust this value in production, or use tracesSampler for greater control.
  tracesSampleRate: 1,
  tracePropagationTargets: [
    "localhost",
    "localhost:8006",
    /^https:\/\/dev\-builder\.agpt\.co\/api/,
    /^https:\/\/.*\.agpt\.co\/api/,
  ],

  // Setting this option to true will print useful information to the console while you're setting up Sentry.
  debug: false,

  // Integrations
  integrations: [
    Sentry.anrIntegration(),
    // NodeProfilingIntegration,
    // Sentry.fsIntegration(),
  ],

  _experiments: {
    // Enable logs to be sent to Sentry.
    enableLogs: true,
  },
});
