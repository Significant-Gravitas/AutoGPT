// This file configures the initialization of Sentry for edge features (middleware, edge routes, and so on).
// The config you add here will be used whenever one of the edge features is loaded.
// Note that this config is unrelated to the Vercel Edge Runtime and is also required when running locally.
// https://docs.sentry.io/platforms/javascript/guides/nextjs/

import * as Sentry from "@sentry/nextjs";
import { getEnvironmentStr } from "./src/lib/utils";

Sentry.init({
  dsn: "https://fe4e4aa4a283391808a5da396da20159@o4505260022104064.ingest.us.sentry.io/4507946746380288",

  environment: getEnvironmentStr(),

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

  _experiments: {
    // Enable logs to be sent to Sentry.
    enableLogs: true,
  },
});
