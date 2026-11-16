// The platform's user-facing workload, weighted like production traffic and
// centred on copilot and experts: opening chats, open chats polling, turns
// on the dummy, scripted and real engines, working with and hiring experts,
// account reads, store browsing, library reads and graph runs watched over
// the WebSocket. See configs/profiles.js for the journeys and their rates,
// configs/thresholds.js for what counts as healthy, and README.md for how to
// run and compare.
//
//   LOAD_PROFILE=smoke ./k6.sh run tests/workload.js
//   LOAD_PROFILE=baseline ./k6.sh run tests/workload.js
//
// setup() signs the run's accounts in, seeds each with graphs, runs, chat
// sessions and a hired expert (lib/seed.js), and probes the copilot engines
// before any turn is sent. teardown() cancels its Stripe test-mode
// subscriptions, deletes the graph runs, library agents and chat sessions it
// created and archives the experts it hired (lib/cleanup.js). The summary
// goes to results/<RUN_ID>.json.
import { Trend } from "k6/metrics";

import {
  buildScenarios,
  durationMs,
  enabledJourneys,
  journeyLoad,
  journeyRate,
  JOURNEYS,
  loadProfile,
} from "../configs/profiles.js";
import {
  buildThresholds,
  customMetricNames,
  steadyKey,
} from "../configs/thresholds.js";
import { accountModeCounts, authSetupTimeout, setupAuth } from "../lib/auth.js";
import { cleanupTimeout, removeRunData } from "../lib/cleanup.js";
import { realTurnCostUsd } from "../lib/copilot.js";
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
  copilotPoll as copilotPollJourney,
  copilotReads as copilotReadsJourney,
  copilotTurn as copilotTurnJourney,
  copilotTurnReal as copilotTurnRealJourney,
  copilotTurnScripted as copilotTurnScriptedJourney,
} from "../scenarios/copilot.js";
import {
  expertHire as expertHireJourney,
  expertWork as expertWorkJourney,
} from "../scenarios/experts.js";
import { graphRun as graphRunJourney } from "../scenarios/graphs.js";
import { libraryBrowse as libraryBrowseJourney } from "../scenarios/library.js";
import { storeBrowse as storeBrowseJourney } from "../scenarios/store.js";

const profile = loadProfile();
const journeys = enabledJourneys();
// How many accounts of each mode the run signs in; reading it here checks
// the mode settings before anything is sent.
const modes = accountModeCounts();
const realTurns = journeys.includes("copilot_turn_real")
  ? JOURNEYS.copilot_turn_real.turns()
  : 0;
const costPerRealTurn = realTurnCostUsd();

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
  const load = {};
  for (const name of journeys) load[name] = journeyLoad(name, profile);
  console.log(
    `workload: profile ${profile.name}, warm-up ${profile.warmup}, steady ${profile.duration}, seed ${runSeed()}; ` +
      `accounts ${modes.dummy} dummy, ${modes.scripted} scripted, ${modes.real} real; load ${JSON.stringify(load)}`,
  );
  if (realTurns > 0) {
    console.log(
      `workload: copilot_turn_real sends ${realTurns} turn(s) to the real model, estimated cost ` +
        `${realTurns} x $${costPerRealTurn.toFixed(2)} = $${(realTurns * costPerRealTurn).toFixed(2)} ` +
        "(COPILOT_REAL_COST_PER_TURN_USD to adjust)",
    );
  }
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

export function copilotPoll(data) {
  copilotPollJourney(data, profile);
}

export function copilotTurn(data) {
  copilotTurnJourney(data, profile);
}

export function copilotTurnScripted(data) {
  copilotTurnScriptedJourney(data, profile);
}

export function copilotTurnReal(data) {
  copilotTurnRealJourney(data, profile);
}

export function expertWork(data) {
  expertWorkJourney(data, profile);
}

export function expertHire(data) {
  expertHireJourney(data, profile);
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

// The real-model turns' cost: the planned turns, the turns the steady phase
// sent, and the estimate for those.
function realTurnCost(summaryData) {
  if (realTurns === 0) return null;
  const sent = summaryData.metrics[steadyKey("copilot_turns{engine:real}")];
  const count = sent && sent.values ? sent.values.count || 0 : 0;
  return {
    real_turns_planned: realTurns,
    real_turns_sent: count,
    per_turn_usd: costPerRealTurn,
    estimated_usd: Math.round(count * costPerRealTurn * 100) / 100,
  };
}

export function handleSummary(data) {
  const scenarios = {};
  for (const name of journeys) {
    const journey = JOURNEYS[name];
    scenarios[name] = journey.turns
      ? { turns: journey.turns() }
      : { rate_per_minute: Math.round(journeyRate(name, profile) * 10) / 10 };
    scenarios[name].requests_per_iteration = journey.requests;
    scenarios[name].why = journey.why;
  }
  return summaryOutputs(data, {
    profile: profile.name,
    seed: runSeed(),
    phases: {
      warmupMs: durationMs(profile.warmup),
      steadyMs: durationMs(profile.duration),
    },
    scenarios,
    customMetrics: customMetricNames(),
    seeding: seedingReport,
    coverage: (summaryData) => coverageReport(summaryData, journeys),
    cost: realTurnCost(data),
  });
}
