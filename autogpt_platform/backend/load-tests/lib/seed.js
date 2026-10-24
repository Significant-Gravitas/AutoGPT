// What a run creates under its own accounts before the load starts, so the
// journeys read and run data that exists and costs that grow with data show
// up. Everything is scoped to the run's accounts: graphs named after the
// run, runs of them, chat sessions, and one marketplace agent added to each
// library. The target's own store listings are discovered, not created.
// teardown() removes what the API lets it remove (lib/cleanup.js).
//
// A new account on a target with credits enabled has none, and a graph run
// needs some; the one-time onboarding rewards fund each account first. On a
// target that paywalls new accounts, STRIPE_TEST_SECRET_KEY lets the seed
// subscribe each account in Stripe test mode before anything else
// (lib/stripe.js); teardown cancels those subscriptions.
//
// What one account cannot get (a paywall, no credits, a failed request)
// keeps that account out of the journey that needs it; the other accounts
// still run it, and the summary's coverage says how many could
// (lib/coverage.js).
import { sleep } from "k6";
import http from "k6/http";
import { Counter, Gauge, Trend } from "k6/metrics";

import { getEnvironmentConfig } from "../configs/environment.js";
import { accountCount, accountHeaders } from "./auth.js";
import { removeRunData } from "./cleanup.js";
import { DUMMY_REPLY, streamTurn } from "./copilot.js";
import {
  COVERED,
  ENGINE_NOT_DUMMY,
  ENGINE_PROBE_FAILED,
  ENGINE_UNSET,
  MARKETPLACE_EMPTY,
  NO_CREDITS,
  NO_GRAPHS,
  NO_LIBRARY_AGENTS,
  NO_SEED_RUN_COMPLETED,
  NO_SESSIONS,
  NOT_SEEDED,
  PAYWALLED,
  SKILLS_UNAVAILABLE,
  SUBSCRIBE_FAILED,
  combine,
  isFailure,
  reasonFor,
  recordCoverage,
} from "./coverage.js";
import { seedGraph } from "./graphs.js";
import {
  cancelSubscriptions,
  stripeTestKey,
  subscribeAccount,
} from "./stripe.js";

const config = getEnvironmentConfig();
const API = config.API_BASE_URL;

export const SEED_GRAPHS = whole("SEED_GRAPHS", 4, 0, 20);
// At least one: the graph journey runs only once a seed run has completed.
export const SEED_RUNS = whole("SEED_RUNS", 2, 1, 10);
export const SEED_SESSIONS = whole("SEED_SESSIONS", 6, 0, 50);
const SEED_TURNS_PER_ACCOUNT = 3;
const SEED_RUN_WAIT_MS = 90000;
const PROBE_TIMEOUT = "60s";

const seedGraphs = new Counter("seed_graphs");
const seedRuns = new Counter("seed_runs");
const seedRunsCompleted = new Counter("seed_runs_completed");
const seedSessions = new Counter("seed_sessions");
const seedTurns = new Counter("seed_turns");
const seedLibraryAdds = new Counter("seed_library_adds");
const seedCredits = new Trend("seed_credits_per_account");
const storeAgentsFound = new Gauge("seed_store_agents_found");
const storeSkillsFound = new Gauge("seed_store_skills_found");

function whole(variable, fallback, min, max) {
  const raw = __ENV[variable];
  if (raw === undefined || raw === "") return fallback;
  const n = Number(raw);
  if (!Number.isInteger(n) || n < min || n > max) {
    throw new Error(
      `${variable} must be a whole number from ${min} to ${max}, got "${raw}"`,
    );
  }
  return n;
}

// For options.setupTimeout: a request per seeded object, the subscription
// calls, a wait for the seed runs, and a probe turn.
const SUBSCRIBE_REQUESTS = 8;

export function seedSetupTimeout() {
  const perAccount =
    2 +
    SUBSCRIBE_REQUESTS +
    SEED_GRAPHS * (2 + SEED_RUNS) +
    SEED_SESSIONS +
    SEED_TURNS_PER_ACCOUNT;
  return `${120 + Math.ceil((accountCount() * perAccount * 1500 + SEED_RUN_WAIT_MS) / 1000)}s`;
}

function request(method, path, headers, body, expected) {
  const res = http.request(
    method,
    `${API}${path}`,
    body === undefined ? null : JSON.stringify(body),
    {
      headers,
      tags: {
        name: `/api${path.split("?")[0].replace(/[0-9a-f-]{36}/g, "{id}")}`,
        phase: "setup",
      },
      responseCallback: http.expectedStatuses(...(expected || [200])),
      timeout: "60s",
    },
  );
  let json = null;
  try {
    json = res.json();
  } catch (e) {
    json = null;
  }
  return {
    res,
    status: res.status,
    ok: (expected || [200]).includes(res.status),
    json,
  };
}

function detail(result) {
  const body = result.json;
  if (!body) return `HTTP ${result.status}`;
  const text = body.detail || body.message || "";
  return `HTTP ${result.status}${text ? ` ${String(text).slice(0, 120)}` : ""}`;
}

// A 402 is the paywall when it asks for a subscription, and a lack of
// credits otherwise. A paywall that holds although the run has a Stripe key
// means subscribing that account failed.
function refusal(result, data) {
  if (!/subscription/i.test(detail(result))) return NO_CREDITS;
  return data.stripe.enabled ? SUBSCRIBE_FAILED : PAYWALLED;
}

export function seedRun(auth, journeys) {
  const wants = (name) => journeys.includes(name);
  const runId = __ENV.RUN_ID || "manual";
  if (wants("copilot_turn")) copilotEngine();
  const data = {
    auth: auth.auth,
    accounts: [],
    store: discoverStore(),
    graphs: { enabled: false, code: COVERED, reason: "not in SCENARIOS" },
    copilot: {
      enabled: false,
      code: COVERED,
      reason: "not in SCENARIOS",
      turnTimeout: __ENV.COPILOT_TURN_TIMEOUT || "120s",
    },
    stripe: {
      enabled: stripeTestKey(config) !== null,
      subscribed: 0,
      failed: 0,
      tier: null,
      subscriptions: [],
    },
  };

  try {
    seedAccounts(auth, wants, data, runId);
  } catch (e) {
    // k6 skips teardown() when setup() throws, so what the seed made so far
    // is removed here, the subscriptions first, as teardown() does.
    try {
      cancelRunSubscriptions(data, "setup failed");
    } finally {
      removeRunData(data, "setup failed");
    }
    throw e;
  }

  const { off, failed } = recordRunCoverage(data, journeys);
  const subscriptions = data.stripe.enabled
    ? `${data.stripe.subscribed} of ${data.accounts.length} subscribed in Stripe test mode${
        data.stripe.tier ? ` (${data.stripe.tier})` : ""
      }`
    : "none (STRIPE_TEST_SECRET_KEY is not set)";
  const line = `seed: ${data.accounts.length} accounts; subscriptions ${subscriptions}; store agents ${data.store.agents.length}, skills ${
    data.store.skillsAvailable ? data.store.skillSlugs.length : "unavailable"
  }; graph runs ${
    data.graphs.enabled
      ? `on for ${able(data, "graphRuns")} of ${data.accounts.length} accounts`
      : `off (${data.graphs.reason})`
  }; copilot turns ${
    data.copilot.enabled
      ? `on for ${able(data, "chat")} of ${data.accounts.length} accounts (dummy engine confirmed)`
      : `off (${data.copilot.reason})`
  }`;
  if (failed.length > 0) {
    console.error(
      `${line}. FAILED ON THE TARGET: ${failed.join(", ")}, which fails the run; NOT COVERED IN FULL: ${off.join(", ")}; the summary's coverage section says why`,
    );
  } else if (off.length > 0) {
    console.warn(
      `${line}. NOT COVERED IN FULL: ${off.join(", ")}; the summary's coverage section says why`,
    );
  } else {
    console.log(line);
  }
  return data;
}

// Subscribes and seeds every account, then decides which journeys can run.
function seedAccounts(auth, wants, data, runId) {
  const pendingRuns = [];
  for (let i = 0; i < auth.auth.length; i++) {
    const headers = accountHeaders(auth, i);
    const account = {
      index: i,
      graphs: [],
      libraryAgentIds: [],
      sessionIds: [],
      // COVERED, or why this account cannot run graphs or chat: a code from
      // lib/coverage.js.
      graphRuns: COVERED,
      chat: COVERED,
    };
    data.accounts.push(account);

    // Before any request that would meet the paywall, and before any that
    // reaches the backend's once-per-five-minutes Stripe reconcile.
    if (data.stripe.enabled) {
      const sub = subscribeAccount(config, headers, runId);
      if (sub.subscriptionId)
        data.stripe.subscriptions.push(sub.subscriptionId);
      if (sub.ok) {
        data.stripe.subscribed++;
        data.stripe.tier = sub.tier;
      } else {
        data.stripe.failed++;
        console.warn(
          `seed: subscribing account ${i + 1} in Stripe test mode failed: ${sub.reason}`,
        );
      }
    }

    if (wants("graph_run") || wants("library_browse")) {
      seedGraphsAndLibrary(account, headers, data, runId);
    }
    if (wants("graph_run")) {
      startSeedRuns(account, headers, data, pendingRuns);
    }
    if (wants("copilot_reads") || wants("copilot_turn")) {
      seedChatSessions(account, headers, data);
    }
  }

  if (wants("graph_run")) decideGraphRuns(data, pendingRuns);
  if (wants("copilot_turn")) decideCopilot(auth, data);
}

function able(data, field) {
  return data.accounts.filter((account) => account[field] === COVERED).length;
}

// Graphs from deterministic blocks, each also a library agent, plus one
// marketplace agent in the library when the store has one.
function seedGraphsAndLibrary(account, headers, data, runId) {
  const number = account.index + 1;
  fundAccount(headers);
  for (let g = 0; g < SEED_GRAPHS; g++) {
    const spec = seedGraph(runId, number, g);
    const created = request("POST", "/graphs", headers, spec.body);
    if (!created.ok) {
      console.warn(
        `seed: creating graph ${g + 1} for account ${number} failed: ${detail(created)}`,
      );
      continue;
    }
    seedGraphs.add(1);
    const graph = {
      graphId: created.json.id,
      version: created.json.version,
      inputs: spec.inputs,
      libraryAgentId: null,
    };
    const library = request(
      "GET",
      `/library/agents/by-graph/${graph.graphId}`,
      headers,
    );
    if (library.ok && library.json && library.json.id) {
      graph.libraryAgentId = library.json.id;
      account.libraryAgentIds.push(library.json.id);
    } else {
      console.warn(
        `seed: graph ${graph.graphId} has no library agent (${detail(library)})`,
      );
    }
    account.graphs.push(graph);
  }
  if (data.store.listingVersionId) {
    const added = request(
      "POST",
      "/library/agents",
      headers,
      {
        store_listing_version_id: data.store.listingVersionId,
        source: "marketplace",
      },
      [200, 201],
    );
    if (added.ok && added.json && added.json.id) {
      seedLibraryAdds.add(1);
      account.libraryAgentIds.push(added.json.id);
    }
  }
  const credits = request("GET", "/credits", headers);
  if (credits.ok && credits.json)
    seedCredits.add(Number(credits.json.credits) || 0);
}

// SEED_RUNS runs of each of the account's graphs. A 402 keeps this account
// out of the graph journey and leaves the other accounts to try theirs.
function startSeedRuns(account, headers, data, pendingRuns) {
  const number = account.index + 1;
  if (account.graphs.length === 0) {
    account.graphRuns = SEED_GRAPHS === 0 ? NOT_SEEDED : NO_GRAPHS;
    return;
  }
  for (const graph of account.graphs) {
    for (let r = 0; r < SEED_RUNS; r++) {
      const started = request(
        "POST",
        `/graphs/${graph.graphId}/execute/${graph.version}`,
        headers,
        { inputs: graph.inputs, credentials_inputs: {}, source: "library" },
        [200, 402],
      );
      if (started.status === 402) {
        account.graphRuns = refusal(started, data);
        console.warn(
          `seed: account ${number} cannot start a run: ${detail(started)}`,
        );
        return;
      }
      if (!started.ok || !started.json || !started.json.id) {
        console.warn(
          `seed: starting a run of ${graph.graphId} failed: ${detail(started)}`,
        );
        continue;
      }
      seedRuns.add(1);
      pendingRuns.push({
        headers,
        graphId: graph.graphId,
        execId: started.json.id,
      });
    }
  }
}

function seedChatSessions(account, headers, data) {
  const number = account.index + 1;
  if (SEED_SESSIONS === 0) {
    account.chat = NOT_SEEDED;
    return;
  }
  for (let s = 0; s < SEED_SESSIONS; s++) {
    const created = request("POST", "/chat/sessions", headers, {}, [200, 402]);
    if (created.status === 402) {
      if (account.sessionIds.length === 0) {
        account.chat = refusal(created, data);
      }
      console.warn(
        `seed: creating chat sessions for account ${number} refused: ${detail(created)}`,
      );
      break;
    }
    if (!created.ok || !created.json || !created.json.id) {
      console.warn(
        `seed: creating a chat session for account ${number} failed: ${detail(created)}`,
      );
      continue;
    }
    seedSessions.add(1);
    account.sessionIds.push(created.json.id);
  }
  if (account.sessionIds.length === 0 && account.chat === COVERED) {
    account.chat = NO_SESSIONS;
  }
}

// The graph journey runs on the accounts that could start their seed runs,
// once one of those runs has completed: a target whose executor completes
// none cannot serve it, and that fails the run.
function decideGraphRuns(data, pendingRuns) {
  if (able(data, "graphRuns") === 0) {
    data.graphs.code = combine(data.accounts.map((a) => a.graphRuns));
    data.graphs.reason = `no account can start a run: ${reasonFor(data.graphs.code)}`;
    return;
  }
  const outcome = waitForRuns(pendingRuns);
  seedRunsCompleted.add(outcome.completed);
  if (outcome.completed === 0) {
    data.graphs.code = NO_SEED_RUN_COMPLETED;
    data.graphs.reason = `none of ${pendingRuns.length} seed runs completed within ${
      SEED_RUN_WAIT_MS / 1000
    }s (${outcome.failed} failed, ${outcome.open} still open): is the executor running?`;
    return;
  }
  if (outcome.failed > 0 || outcome.open > 0) {
    console.warn(
      `seed: ${outcome.completed} of ${pendingRuns.length} seed runs completed, ${outcome.failed} failed, ${outcome.open} still open`,
    );
  }
  data.graphs.enabled = true;
  data.graphs.code = combine(data.accounts.map((a) => a.graphRuns));
  data.graphs.reason = null;
}

// What each enabled journey can exercise, from what the seed found and
// created. Returns the journeys that cannot run in full and those of them
// that failed on the target.
function recordRunCoverage(data, journeys) {
  const library = data.accounts.map(libraryCoverage);
  const entries = {
    account_reads: [COVERED],
    store_browse: [storeCoverage(data.store)],
    library_browse: [
      combine(library),
      library.filter((code) => code === COVERED).length,
    ],
    graph_run: [
      data.graphs.code,
      data.graphs.enabled ? able(data, "graphRuns") : 0,
    ],
    copilot_reads: [
      combine(data.accounts.map((a) => a.chat)),
      able(data, "chat"),
    ],
    copilot_turn: [
      data.copilot.code,
      data.copilot.enabled ? able(data, "chat") : 0,
    ],
  };
  const off = [];
  const failed = [];
  for (const name of journeys) {
    const [code, accounts] = entries[name];
    recordCoverage(name, code, accounts);
    if (code !== COVERED) off.push(name);
    if (isFailure(code)) failed.push(name);
  }
  return { off, failed };
}

function storeCoverage(store) {
  if (store.agents.length === 0) return MARKETPLACE_EMPTY;
  if (!store.skillsAvailable || store.skillSlugs.length === 0) {
    return SKILLS_UNAVAILABLE;
  }
  return COVERED;
}

// The library journey opens one of the account's library agents.
function libraryCoverage(account) {
  if (account.libraryAgentIds.length > 0) return COVERED;
  if (account.graphs.length > 0) return NO_LIBRARY_AGENTS;
  return SEED_GRAPHS === 0 ? NOT_SEEDED : NO_GRAPHS;
}

// For teardown(), and for a setup() that throws: cancels the subscriptions
// the run created.
export function cancelRunSubscriptions(data, when) {
  const ids = (data && data.stripe && data.stripe.subscriptions) || [];
  if (ids.length === 0) return;
  const result = cancelSubscriptions(ids);
  console.log(
    `${when}: cancelled ${result.cancelled} of ${ids.length} Stripe test-mode subscription(s)${
      result.failed > 0
        ? `; ${result.failed} failed, scripts/cancel-load-test-subscriptions.js cancels leftovers`
        : ""
    }`,
  );
}

// The one-time onboarding rewards: 300 credits each for two steps, more
// than enough for a run's worth of zero-cost blocks.
function fundAccount(headers) {
  for (const step of ["AGENT_NEW_RUN", "ONBOARDING_COMPLETE"]) {
    const res = request(
      "POST",
      `/onboarding/step?step=${step}`,
      headers,
      undefined,
      [200, 400, 409],
    );
    if (!res.ok)
      console.warn(`seed: onboarding step ${step} answered ${detail(res)}`);
  }
}

// The target's own marketplace, as a visitor sees it, since that is how the
// store journey reads it: the first page of agents and skills, and one
// listing version to add to libraries. The skills catalogue sits behind a
// feature flag that the backend evaluates for anonymous visitors apart from
// signed-in users.
function discoverStore() {
  const store = {
    agents: [],
    skillSlugs: [],
    skillsAvailable: false,
    listingVersionId: null,
  };
  const agents = request(
    "GET",
    "/store/agents?page=1&page_size=20&sorted_by=runs",
    {},
  );
  if (agents.ok && agents.json && Array.isArray(agents.json.agents)) {
    store.agents = agents.json.agents
      .filter((a) => a.creator && a.slug && a.creator !== "Needs Profile")
      .map((a) => ({ creator: a.creator, slug: a.slug }));
  } else {
    console.warn(`seed: listing store agents failed: ${detail(agents)}`);
  }
  if (store.agents.length > 0) {
    const first = store.agents[0];
    const page = request(
      "GET",
      `/store/agents/${encodeURIComponent(first.creator)}/${encodeURIComponent(first.slug)}`,
      {},
    );
    if (page.ok && page.json)
      store.listingVersionId = page.json.store_listing_version_id || null;
  } else {
    console.warn(
      "seed: the target's store has no agents; the store journey reads empty lists",
    );
  }
  storeAgentsFound.add(store.agents.length);
  const skills = request(
    "GET",
    "/store/skills?page=1&page_size=50",
    {},
    undefined,
    [200, 404],
  );
  if (
    skills.status === 200 &&
    skills.json &&
    Array.isArray(skills.json.skills)
  ) {
    store.skillsAvailable = true;
    store.skillSlugs = skills.json.skills.map((s) => s.slug).filter(Boolean);
    if (store.skillSlugs.length === 0) {
      console.warn(
        "seed: the target's skills catalogue is empty; the store journey reads the list only",
      );
    }
  } else {
    console.warn(
      `seed: skills are not available to visitors on the target (${detail(skills)}); the store journey skips them`,
    );
  }
  storeSkillsFound.add(store.skillSlugs.length);
  return store;
}

// Polls the seed runs until they finish or the wait runs out.
function waitForRuns(pending) {
  const deadline = Date.now() + SEED_RUN_WAIT_MS;
  let open = pending.slice();
  let completed = 0;
  let failed = 0;
  while (open.length > 0 && Date.now() < deadline) {
    sleep(2);
    const still = [];
    for (const run of open) {
      const res = request(
        "GET",
        `/graphs/${run.graphId}/executions/${run.execId}`,
        run.headers,
      );
      const status = res.json && res.json.status;
      if (status === "COMPLETED") completed++;
      else if (status === "FAILED" || status === "TERMINATED") failed++;
      else still.push(run);
    }
    open = still;
  }
  return { completed, failed, open: open.length };
}

// LOAD_TEST_COPILOT_ENGINE, "" or "dummy". seedRun() checks it before it
// creates anything.
function copilotEngine() {
  const engine = __ENV.LOAD_TEST_COPILOT_ENGINE || "";
  if (engine !== "" && engine !== "dummy") {
    throw new Error(
      `LOAD_TEST_COPILOT_ENGINE=${engine}: the workload sends turns only at the dummy engine; tests/copilot-real-model.js is the opt-in real-model test`,
    );
  }
  return engine;
}

// Copilot turns are sent only when the operator says the target runs the
// dummy engine, and after one probe turn confirms it. A probe that fails
// then is a failure of the target, not a limit.
function decideCopilot(auth, data) {
  if (copilotEngine() === "") {
    data.copilot.code = ENGINE_UNSET;
    data.copilot.reason =
      "LOAD_TEST_COPILOT_ENGINE is not set; the turn journey is opt-in for a target that runs the dummy engine, CHAT_TEST_MODE=true (README: Copilot and model spend)";
    return;
  }
  const first = data.accounts.find((a) => a.chat === COVERED);
  if (!first) {
    data.copilot.code = combine(data.accounts.map((a) => a.chat));
    data.copilot.reason = `no account has a chat session to probe the engine with: ${reasonFor(data.copilot.code)}`;
    return;
  }
  const probe = streamTurn(first.sessionIds[0], "Reply with one word.", {
    headers: accountHeaders(auth, first.index),
    phase: "setup",
    timeout: PROBE_TIMEOUT,
    record: false,
  });
  if (probe.outcome !== "completed") {
    data.copilot.code = ENGINE_PROBE_FAILED;
    data.copilot.reason = `the engine probe ${probe.outcome} (${probe.error || "no error"}); is the copilot executor running?`;
    return;
  }
  if (probe.text.trim() !== DUMMY_REPLY) {
    data.copilot.code = ENGINE_NOT_DUMMY;
    data.copilot.reason =
      "the engine probe's reply was not the dummy's: the target is not in CHAT_TEST_MODE, and the probe spent one real turn";
    return;
  }
  data.copilot.enabled = true;
  data.copilot.code = combine(data.accounts.map((a) => a.chat));
  data.copilot.reason = null;

  // A little history in the seeded sessions, so session reads carry messages.
  for (const account of data.accounts) {
    if (account.chat !== COVERED) continue;
    const headers = accountHeaders(auth, account.index);
    for (const sessionId of account.sessionIds.slice(
      0,
      SEED_TURNS_PER_ACCOUNT,
    )) {
      const turn = streamTurn(sessionId, "Say hello in one word.", {
        headers,
        phase: "setup",
        timeout: PROBE_TIMEOUT,
        record: false,
      });
      if (turn.outcome === "completed") seedTurns.add(1);
    }
  }
}

// For handleSummary: what the seed created and teardown removed, from k6's
// metrics.
export function seedingReport(summaryData) {
  const count = (name) => {
    const metric = summaryData.metrics[name];
    return metric && metric.values ? metric.values.count || 0 : 0;
  };
  const gauge = (name) => {
    const metric = summaryData.metrics[name];
    return metric && metric.values && typeof metric.values.value === "number"
      ? metric.values.value
      : null;
  };
  const credits = summaryData.metrics.seed_credits_per_account;
  return {
    graphs: count("seed_graphs"),
    runs: count("seed_runs"),
    runs_completed: count("seed_runs_completed"),
    sessions: count("seed_sessions"),
    turns: count("seed_turns"),
    library_adds: count("seed_library_adds"),
    subscriptions: count("seed_subscriptions"),
    subscriptions_cancelled: count("seed_subscriptions_cancelled"),
    store_agents_found: gauge("seed_store_agents_found"),
    store_skills_found: gauge("seed_store_skills_found"),
    credits_per_account_avg:
      credits && credits.values ? Math.round(credits.values.avg) : null,
    executions_deleted: count("cleanup_executions"),
    library_agents_deleted: count("cleanup_library_agents"),
    cleanup_failures: count("cleanup_failures"),
  };
}
