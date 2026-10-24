// The platform's user-facing workload, weighted like production traffic:
// copilot reads and turns, account reads, store browsing, library reads and
// graph runs watched over the WebSocket. See configs/profiles.js for the
// journeys and their rates, configs/thresholds.js for what counts as
// healthy, and README.md for how to run and compare.
//
//   LOAD_PROFILE=smoke ./k6.sh run tests/workload.js
//   LOAD_PROFILE=baseline ./k6.sh run tests/workload.js
//
// setup() signs the run's accounts in, seeds each with graphs, runs and chat
// sessions (lib/seed.js), and confirms the copilot engine before any turn is
// sent. teardown() cancels its Stripe test-mode subscriptions and deletes
// the graph runs and library agents it created (lib/cleanup.js). The
// summary goes to results/<RUN_ID>.json.
import { Trend } from "k6/metrics";

import {
  buildScenarios,
  durationMs,
  enabledJourneys,
  journeyRate,
  JOURNEYS,
  loadProfile,
} from "../configs/profiles.js";
import { buildThresholds, customMetricNames } from "../configs/thresholds.js";
import { authSetupTimeout, setupAuth } from "../lib/auth.js";
import { cleanupTimeout, removeRunData } from "../lib/cleanup.js";
import { coverageReport } from "../lib/coverage.js";
import { runSeed } from "../lib/random.js";
import {
  cancelRunSubscriptions,
  seedRun,
  seedSetupTimeout,
  seedingReport,
} from "../lib/seed.js";
import { summaryOutputs } from "../lib/summary.js";
import { accountReads as accountReadsJourney } from "../scenarios/account.js";
import {
  copilotReads as copilotReadsJourney,
  copilotTurn as copilotTurnJourney,
} from "../scenarios/copilot.js";
import { graphRun as graphRunJourney } from "../scenarios/graphs.js";
import { libraryBrowse as libraryBrowseJourney } from "../scenarios/library.js";
import { storeBrowse as storeBrowseJourney } from "../scenarios/store.js";

const profile = loadProfile();
const journeys = enabledJourneys();

const setupDuration = new Trend("setup_duration_ms");

export const options = {
  scenarios: buildScenarios(profile, journeys),
  thresholds: buildThresholds(profile, journeys),
  setupTimeout: `${
    Number.parseInt(authSetupTimeout(), 10) +
    Number.parseInt(seedSetupTimeout(), 10)
  }s`,
  teardownTimeout: cleanupTimeout(),
  summaryTrendStats: [
    "count",
    "avg",
    "min",
    "med",
    "p(90)",
    "p(95)",
    "p(99)",
    "max",
  ],
  userAgent: "autogpt-load-tests/2 (workload)",
};

export function setup() {
  const started = Date.now();
  const rates = {};
  for (const name of journeys) {
    rates[name] = `${journeyRate(name, profile).toFixed(1)}/min`;
  }
  console.log(
    `workload: profile ${profile.name}, warm-up ${profile.warmup}, steady ${profile.duration}, seed ${runSeed()}; ` +
      `rates ${JSON.stringify(rates)}`,
  );
  const auth = setupAuth();
  const data = seedRun(auth, journeys);
  setupDuration.add(Date.now() - started);
  console.log(
    `workload: setup took ${Math.round((Date.now() - started) / 1000)}s`,
  );
  return data;
}

// The subscriptions first: the deletes need none, and a teardown that runs
// out of time on a struggling target must not leave them active.
export function teardown(data) {
  try {
    cancelRunSubscriptions(data, "teardown");
  } finally {
    removeRunData(data, "teardown");
  }
}

export function copilotReads(data) {
  copilotReadsJourney(data, profile);
}

export function copilotTurn(data) {
  copilotTurnJourney(data, profile);
}

export function accountReads(data) {
  accountReadsJourney(data, profile);
}

export function storeBrowse(data) {
  storeBrowseJourney(data, profile);
}

export function libraryBrowse(data) {
  libraryBrowseJourney(data, profile);
}

export function graphRun(data) {
  graphRunJourney(data, profile);
}

export function handleSummary(data) {
  const scenarios = {};
  for (const name of journeys) {
    scenarios[name] = {
      rate_per_minute: Math.round(journeyRate(name, profile) * 10) / 10,
      requests_per_iteration: JOURNEYS[name].requests,
      why: JOURNEYS[name].why,
    };
  }
  return summaryOutputs(data, {
    profile: profile.name,
    seed: runSeed(),
    phases: {
      warmupMs: durationMs(profile.warmup),
      steadyMs: durationMs(profile.duration),
    },
    scenarios,
    customMetrics: customMetricNames().concat([
      "copilot_chunks_per_turn",
      "copilot_turns",
      "copilot_turns_queued",
      "copilot_turns_failed",
      "graph_runs",
      "graph_runs_failed",
      "ws_events",
    ]),
    seeding: seedingReport,
    coverage: (summaryData) => coverageReport(summaryData, journeys),
  });
}
