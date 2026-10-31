// What each journey could exercise on the target. setup() records it and the
// summary reports it, so a run that measured less than it was asked to says
// so instead of showing fast numbers for empty lists and skipped scenarios.
//
// Each code is a limit or a failure. A limit is the target's configuration
// or data, or the operator's choice: a paywall with no Stripe key, an empty
// marketplace, a copilot engine nobody asked for. The journey runs what it
// can or skips, and the run passes or fails on what it measured. A failure
// is the target not doing what the run needs and asked for: no seed run
// completes, a requested engine fails its probe, the seed cannot create what
// a journey reads. It fails the run through coverage_failures, whose
// threshold is in configs/thresholds.js, because the journey's own
// thresholds have no samples to fail on.
//
// setup() cannot hand text to handleSummary, so the state travels as
// numbers: per journey, a code and how many accounts can run the journey.
// The log carries the detail behind each code.
import { Counter, Gauge } from "k6/metrics";

import { JOURNEYS } from "../configs/profiles.js";
import { accountCount } from "./auth.js";

export const COVERED = 0;
export const PAYWALLED = 1;
export const ENGINE_UNSET = 2;
export const ENGINE_PROBE_FAILED = 3;
export const ENGINE_NOT_DUMMY = 4;
export const NO_SEED_RUN_COMPLETED = 5;
export const MARKETPLACE_EMPTY = 6;
export const SKILLS_UNAVAILABLE = 7;
export const NO_SESSIONS = 8;
export const NO_GRAPHS = 9;
export const NO_CREDITS = 10;
export const SUBSCRIBE_FAILED = 11;
export const NOT_SEEDED = 12;
export const NO_LIBRARY_AGENTS = 13;

const CODES = {
  [PAYWALLED]: {
    failure: false,
    reason:
      "the target paywalls new accounts (402 A subscription is required); STRIPE_TEST_SECRET_KEY lets the run subscribe its accounts in Stripe test mode (README: Paywalled targets)",
  },
  [ENGINE_UNSET]: {
    failure: false,
    reason:
      "LOAD_TEST_COPILOT_ENGINE is not set; see README, Copilot and model spend",
  },
  [MARKETPLACE_EMPTY]: {
    failure: false,
    reason:
      "the target's marketplace is empty: the lists answered with no items and no detail page was opened",
  },
  [SKILLS_UNAVAILABLE]: {
    failure: false,
    reason:
      "the target's skills catalogue is unavailable to visitors or empty: no skill detail was opened",
  },
  [NO_CREDITS]: {
    failure: false,
    reason: "an account has no credits left (402)",
  },
  [NOT_SEEDED]: {
    failure: false,
    reason:
      "SEED_GRAPHS or SEED_SESSIONS is 0, so the seed created nothing for the journey to open",
  },
  [ENGINE_PROBE_FAILED]: {
    failure: true,
    reason:
      "the copilot engine probe did not complete; is the copilot executor running?",
  },
  [ENGINE_NOT_DUMMY]: {
    failure: true,
    reason:
      "LOAD_TEST_COPILOT_ENGINE=dummy, but the engine probe's reply was not the dummy's: the target is not in CHAT_TEST_MODE",
  },
  [NO_SEED_RUN_COMPLETED]: {
    failure: true,
    reason:
      "no seed run completed, so the graph journey did not run; is the executor running?",
  },
  [NO_SESSIONS]: {
    failure: true,
    reason:
      "the seed could not create a chat session: the session list is empty and no session detail was read",
  },
  [NO_GRAPHS]: {
    failure: true,
    reason: "the seed could not create a graph",
  },
  [SUBSCRIBE_FAILED]: {
    failure: true,
    reason:
      "an account stayed paywalled although the run had a Stripe key: subscribing it failed or the target did not honour the subscription; the log says why",
  },
  [NO_LIBRARY_AGENTS]: {
    failure: true,
    reason:
      "the seeded graphs have no library agents, so no library agent detail was read",
  },
};

export function isFailure(code) {
  return Boolean(CODES[code] && CODES[code].failure);
}

export function reasonFor(code) {
  return CODES[code] ? CODES[code].reason : `code ${code}`;
}

// One journey's code from its accounts' codes: a failure on any account
// wins, then a limit on any account, else covered.
export function combine(codes) {
  const failure = codes.find((code) => isFailure(code));
  if (failure !== undefined) return failure;
  const limit = codes.find((code) => code !== COVERED);
  return limit === undefined ? COVERED : limit;
}

// A code leaves these journeys with nothing to do; the others still make
// the reads that do not depend on what is missing.
const SKIPPED_ENTIRELY = { graph_run: true, copilot_turn: true };

const codeGauges = {};
const accountGauges = {};
for (const name of Object.keys(JOURNEYS)) {
  codeGauges[name] = new Gauge(`coverage_${name}`);
  accountGauges[name] = new Gauge(`coverage_${name}_accounts`);
}
const coverageFailures = new Counter("coverage_failures");

// `accounts` is how many of the run's accounts can run the journey in full;
// leave it out for a journey that does not depend on the account.
export function recordCoverage(name, code, accounts) {
  codeGauges[name].add(code);
  if (accounts !== undefined) accountGauges[name].add(accounts);
  if (isFailure(code)) coverageFailures.add(1, { journey: name });
}

// For handleSummary: per journey, "covered", "partial", "skipped" or
// "failed", why, and how many accounts could run it.
export function coverageReport(summaryData, journeys) {
  const total = accountCount();
  const report = {};
  for (const name of journeys) {
    const code = gaugeValue(summaryData, `coverage_${name}`);
    if (code === null) {
      report[name] = {
        status: "unknown",
        reason: "setup() recorded no coverage for this journey",
      };
      continue;
    }
    const able = gaugeValue(summaryData, `coverage_${name}_accounts`);
    const entry = { status: "covered", reason: null };
    if (code !== COVERED) {
      entry.reason = reasonFor(code);
      if (isFailure(code)) entry.status = "failed";
      else if (SKIPPED_ENTIRELY[name] && !able) entry.status = "skipped";
      else entry.status = "partial";
    }
    if (able !== null) entry.accounts = `${able} of ${total}`;
    report[name] = entry;
  }
  return report;
}

function gaugeValue(summaryData, name) {
  const metric = summaryData.metrics[name];
  return metric && metric.values && typeof metric.values.value === "number"
    ? metric.values.value
    : null;
}
