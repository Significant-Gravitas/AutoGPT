// A few copilot turns against the target's real engine, to measure what a
// user feels with a real model behind the chat, on its own rather than
// within the workload's mix (whose copilot_turn_real sends a few too). It
// spends money, so it runs only with LOAD_TEST_COPILOT_ENGINE=real, sends a
// capped number of turns (COPILOT_REAL_TURNS, default 5) one or a few at a
// time (COPILOT_REAL_CONCURRENCY, default 1), and reports the cost: an
// estimate from COPILOT_REAL_COST_PER_TURN_USD (default $0.07, production's
// median one-step turn) and how far the account's daily usage meter moved.
// Its accounts are all real-model ones (README: Account modes).
//
//   LOAD_TEST_COPILOT_ENGINE=real COPILOT_REAL_TURNS=5 ./k6.sh run tests/copilot-real-model.js
import { Trend } from "k6/metrics";

import { getEnvironmentConfig } from "../configs/environment.js";
import { authSetupTimeout, setupAuth } from "../lib/auth.js";
import {
  realTurnCostUsd,
  recordUnstartedTurn,
  streamTurn,
} from "../lib/copilot.js";
import { api } from "../lib/http.js";
import { seededRandom, pick, runSeed } from "../lib/random.js";
import { summaryOutputs } from "../lib/summary.js";
import { MESSAGES } from "../scenarios/copilot.js";

const config = getEnvironmentConfig();

const TURNS = wholeNumber("COPILOT_REAL_TURNS", 5, 1, 50);
const CONCURRENCY = wholeNumber("COPILOT_REAL_CONCURRENCY", 1, 1, 5);

// A whole number from the environment, `fallback` when unset; anything else
// out of [min, max] stops the run at init.
function wholeNumber(name, fallback, min, max) {
  const raw = __ENV[name];
  if (raw === undefined || raw === "") return fallback;
  const value = Number(raw);
  if (!Number.isInteger(value) || value < min || value > max) {
    throw new Error(
      `${name} must be a whole number from ${min} to ${max}, got "${raw}"`,
    );
  }
  return value;
}
const COST_PER_TURN_USD = realTurnCostUsd();
const TURN_TIMEOUT = __ENV.COPILOT_TURN_TIMEOUT || "300s";

const setupDuration = new Trend("setup_duration_ms");
const usagePercent = new Trend("copilot_usage_percent_of_daily_cap");

export const options = {
  scenarios: {
    copilot_real: {
      executor: "shared-iterations",
      exec: "realTurn",
      vus: CONCURRENCY,
      iterations: TURNS,
      maxDuration: "30m",
      tags: { journey: "copilot_real" },
    },
  },
  thresholds: {
    "copilot_turn_errors{phase:steady}": ["rate<0.2"],
  },
  setupTimeout: authSetupTimeout(),
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
  userAgent: "autogpt-load-tests/2 (copilot-real-model)",
};

export function setup() {
  if (__ENV.LOAD_TEST_COPILOT_ENGINE !== "real") {
    throw new Error(
      "This test sends turns to the target's real model and costs money. " +
        "Set LOAD_TEST_COPILOT_ENGINE=real to run it, and COPILOT_REAL_TURNS to cap the turns (default 5).",
    );
  }
  console.log(
    `copilot-real-model: ${TURNS} turn(s), ${CONCURRENCY} at a time, at ${config.NAME}; ` +
      `estimated cost ${TURNS} x $${COST_PER_TURN_USD.toFixed(2)} = $${(
        TURNS * COST_PER_TURN_USD
      ).toFixed(2)} (COPILOT_REAL_COST_PER_TURN_USD to adjust)`,
  );
  const started = Date.now();
  // One account per VU unless LOAD_TEST_ACCOUNT_COUNT says otherwise: the
  // turns go one at a time per VU, so more accounts would sit idle.
  const data = setupAuth({ mode: "real", defaultCount: CONCURRENCY });
  const usage = api("GET", "/chat/usage", {
    data,
    endpoint: "copilot.usage",
    phase: "warmup",
  });
  data.usageBefore = usage.json;
  setupDuration.add(Date.now() - started);
  return data;
}

export function realTurn(data) {
  const random = seededRandom(`${runSeed()}:copilot_real:${__ITER}:${__VU}`);
  const created = api("POST", "/chat/sessions", {
    data,
    body: {},
    endpoint: "copilot.session_create",
    phase: "steady",
  });
  const sessionId = created.json && created.json.id;
  if (!sessionId) {
    if (created.status === 402 && __ITER === 0) {
      console.warn(
        "copilot_real: the target asks this account for a subscription (402), and this test does not subscribe its accounts. " +
          "On a paywalled preview run tests/workload.js with LOAD_TEST_COPILOT_REAL_TURNS and STRIPE_TEST_SECRET_KEY instead (README: Real-model turns).",
      );
    }
    // A turn that could not start is a failed turn, so a run that sends
    // none fails copilot_turn_errors instead of passing with no samples.
    recordUnstartedTurn("real", "steady");
    return;
  }
  const turn = streamTurn(sessionId, pick(random, MESSAGES), {
    data,
    phase: "steady",
    engine: "real",
    timeout: TURN_TIMEOUT,
  });
  console.log(
    `turn ${turn.outcome}: first chunk ${turn.firstChunkMs}ms, first token ${turn.ttftMs}ms, ` +
      `done ${turn.turnMs}ms, ${turn.chunks} frames, ${turn.text.length} chars`,
  );
}

export function teardown(data) {
  const usage = api("GET", "/chat/usage", {
    data,
    endpoint: "copilot.usage",
    phase: "warmup",
  });
  const before = dailyPercent(data.usageBefore);
  const after = dailyPercent(usage.json);
  if (before !== null && after !== null) {
    usagePercent.add(after - before);
    console.log(
      `copilot-real-model: the account's daily usage meter moved from ${before}% to ${after}% of its cap`,
    );
  }
  console.log(
    `copilot-real-model: estimated spend $${(TURNS * COST_PER_TURN_USD).toFixed(2)} for ${TURNS} turn(s)`,
  );
}

function dailyPercent(usage) {
  if (!usage || !usage.daily || typeof usage.daily.percent_used !== "number") {
    return null;
  }
  return usage.daily.percent_used;
}

export function handleSummary(data) {
  return summaryOutputs(data, {
    profile: "copilot-real-model",
    seed: runSeed(),
    scenarios: {
      copilot_real: { turns: TURNS, concurrency: CONCURRENCY },
    },
    customMetrics: [
      "copilot_first_chunk_ms",
      "copilot_ttft_ms",
      "copilot_turn_ms",
      "copilot_chunks_per_turn",
      "copilot_turn_errors",
      "copilot_usage_percent_of_daily_cap",
    ],
    cost: {
      estimated_usd: TURNS * COST_PER_TURN_USD,
      per_turn_usd: COST_PER_TURN_USD,
    },
  });
}
