// What each journey could exercise on the target. setup() records it and the
// summary reports it, so a run that measured less than it was asked to says
// so instead of showing fast numbers for empty lists and skipped scenarios.
//
// setup() cannot hand text to handleSummary, so the state travels as a code
// in one Gauge per journey; the log carries the detail behind each code.
import { Gauge } from "k6/metrics";

import { JOURNEYS } from "../configs/profiles.js";

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

const REASONS = {
  [PAYWALLED]:
    "the target paywalls new accounts (402 A subscription is required); see README, Paywalled targets",
  [ENGINE_UNSET]:
    "LOAD_TEST_COPILOT_ENGINE is not set; see README, Copilot and model spend",
  [ENGINE_PROBE_FAILED]:
    "the copilot engine probe did not complete; is the copilot executor running?",
  [ENGINE_NOT_DUMMY]:
    "the engine probe's reply was not the dummy's: the target is not in CHAT_TEST_MODE",
  [NO_SEED_RUN_COMPLETED]: "no seed run completed; is the executor running?",
  [MARKETPLACE_EMPTY]:
    "the target's marketplace is empty: the lists answered with no items and no detail page was opened",
  [SKILLS_UNAVAILABLE]:
    "the target's skills catalogue is unavailable or empty: no skill detail was opened",
  [NO_SESSIONS]:
    "no chat session exists for the run's accounts: the session list is empty and no session detail was read",
  [NO_GRAPHS]: "no graph could be seeded",
  [NO_CREDITS]: "the accounts have no credits for a run",
};

// A code leaves these journeys with nothing to do; the others still make
// the reads that do not depend on what is missing.
const SKIPPED_ENTIRELY = { graph_run: true, copilot_turn: true };

const gauges = {};
for (const name of Object.keys(JOURNEYS)) {
  gauges[name] = new Gauge(`coverage_${name}`);
}

export function recordCoverage(name, code) {
  gauges[name].add(code);
}

// For handleSummary: per journey, "covered", "partial" or "skipped" and why.
export function coverageReport(summaryData, journeys) {
  const report = {};
  for (const name of journeys) {
    const metric = summaryData.metrics[`coverage_${name}`];
    if (!metric || !metric.values || typeof metric.values.value !== "number") {
      report[name] = {
        status: "unknown",
        reason: "setup() recorded no coverage for this journey",
      };
      continue;
    }
    const code = metric.values.value;
    if (code === COVERED) {
      report[name] = { status: "covered", reason: null };
    } else {
      report[name] = {
        status: SKIPPED_ENTIRELY[name] ? "skipped" : "partial",
        reason: REASONS[code] || `code ${code}`,
      };
    }
  }
  return report;
}
