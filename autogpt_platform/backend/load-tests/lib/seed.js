// What a run creates under its own accounts before the load starts, so the
// journeys read and run data that exists and costs that grow with data show
// up. Everything is scoped to the run's accounts: graphs named after the
// run, runs of them, chat sessions, one marketplace agent added to each
// library, and a hired expert with two chats for each dummy and scripted
// account. The target's own store listings and expert templates are
// discovered, not created. teardown() removes what the API lets it remove
// (lib/cleanup.js).
//
// A new account on a target with credits enabled has none, and a graph run
// needs some; the one-time onboarding rewards fund each account first. On a
// target that paywalls new accounts, STRIPE_TEST_SECRET_KEY lets the seed
// subscribe each account in Stripe test mode before anything else
// (lib/stripe.js); teardown cancels those subscriptions.
//
// Before a journey sends a copilot turn, a probe confirms the target runs
// each account's engine (lib/auth.js): one dummy account's turn must come
// back with the dummy's reply, and only then, so that a target which does
// not route accounts by address never gets a turn that costs money, one
// scripted account's turn must end with the script's final text.
//
// What one account cannot get (a paywall, no credits, a failed request)
// keeps that account out of the journey that needs it; the other accounts
// still run it, and the summary's coverage says how many could
// (lib/coverage.js).
import { sleep } from "k6";
import http from "k6/http";
import { Counter, Gauge, Trend } from "k6/metrics";

import { getEnvironmentConfig } from "../configs/environment.js";
import { durationMs, TURN_TIMEOUTS } from "../configs/profiles.js";
import { accountCount, accountHeaders } from "./auth.js";
import { removeRunData } from "./cleanup.js";
import {
  DUMMY_REPLY,
  SCRIPTED_TOOL_OUTPUT,
  hasScriptedReply,
  scriptedMessage,
  streamTurn,
} from "./copilot.js";
import {
  COVERED,
  ENGINE_NOT_DUMMY,
  ENGINE_PROBE_FAILED,
  ENGINE_UNCONFIRMED,
  MARKETPLACE_EMPTY,
  NO_ACCOUNTS_OF_MODE,
  NO_CREDITS,
  NO_EXPERT,
  NO_EXPERT_TEMPLATES,
  NO_GRAPHS,
  NO_LIBRARY_AGENTS,
  NO_SEED_RUN_COMPLETED,
  NO_SESSIONS,
  NO_SPARE_TEMPLATE,
  NOT_SEEDED,
  PAYWALLED,
  SCRIPTED_PROBE_FAILED,
  SKILLS_UNAVAILABLE,
  STORE_DISCOVERY_FAILED,
  SUBSCRIBE_FAILED,
  TURNS_OFF,
  combine,
  isFailure,
  reasonFor,
  recordCoverage,
} from "./coverage.js";
import { seedGraph } from "./graphs.js";
import { randomUuid } from "./random.js";
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
// Each work expert's chats, which expert_work opens and continues.
const SEED_EXPERT_SESSIONS = 2;
const SEED_RUN_WAIT_MS = 90000;
// A hire answers before its workflows, skills and routines are installed;
// the seed waits for them, every account's at once, up to this long.
const EXPERT_SETUP_WAIT_MS = 120000;
const PROBE_TIMEOUT = "60s";

// The journeys that send copilot turns, each on its accounts' engine.
const TURN_JOURNEYS = {
  copilot_turn: "dummy",
  expert_work: "dummy",
  expert_hire: "dummy",
  copilot_turn_scripted: "scripted",
  copilot_turn_real: "real",
};
// The journeys that read chat sessions or create them.
const CHAT_JOURNEYS = ["copilot_reads", "copilot_poll"].concat(
  Object.keys(TURN_JOURNEYS),
);

const seedGraphs = new Counter("seed_graphs");
const seedRuns = new Counter("seed_runs");
const seedRunsCompleted = new Counter("seed_runs_completed");
const seedSessions = new Counter("seed_sessions");
const seedTurns = new Counter("seed_turns");
const seedLibraryAdds = new Counter("seed_library_adds");
const seedExpertsHired = new Counter("seed_experts");
const seedExpertSessions = new Counter("seed_expert_sessions");
const seedCredits = new Trend("seed_credits_per_account");
const storeAgentsFound = new Gauge("seed_store_agents_found");
const storeSkillsFound = new Gauge("seed_store_skills_found");
const expertTemplatesFound = new Gauge("seed_expert_templates_found");

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

// LOAD_TEST_COPILOT_TURNS=off sends no copilot turn at all, probes included.
function copilotTurnsOff() {
  const raw = (__ENV.LOAD_TEST_COPILOT_TURNS || "").trim().toLowerCase();
  if (raw === "" || raw === "on") return false;
  if (raw === "off") return true;
  throw new Error(`LOAD_TEST_COPILOT_TURNS must be on or off, got "${raw}"`);
}

// For options.setupTimeout: a request per seeded object, the subscription
// calls, the expert's hire and chats, a wait for the seed runs and one for
// the experts' setup, and the two probe turns.
const SUBSCRIBE_REQUESTS = 8;
const EXPERT_REQUESTS = 4 + SEED_EXPERT_SESSIONS;

export function seedSetupTimeout() {
  const perAccount =
    2 +
    SUBSCRIBE_REQUESTS +
    SEED_GRAPHS * (2 + SEED_RUNS) +
    SEED_SESSIONS +
    SEED_TURNS_PER_ACCOUNT +
    EXPERT_REQUESTS;
  const waits =
    SEED_RUN_WAIT_MS +
    EXPERT_SETUP_WAIT_MS +
    durationMs(PROBE_TIMEOUT) +
    durationMs(TURN_TIMEOUTS.scripted);
  return `${120 + Math.ceil((accountCount() * perAccount * 1500 + waits) / 1000)}s`;
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
  const text =
    typeof body.detail === "string"
      ? body.detail
      : (body.detail && (body.detail.code || JSON.stringify(body.detail))) ||
        body.message ||
        "";
  return `HTTP ${result.status}${text ? ` ${String(text).slice(0, 120)}` : ""}`;
}

// A 402 is the paywall when it asks for a subscription, and a lack of
// credits otherwise. A paywall that holds although the run has a Stripe key
// means subscribing that account failed.
function refusal(result, data) {
  if (!/subscription/i.test(detail(result))) return NO_CREDITS;
  return data.stripe.enabled ? SUBSCRIBE_FAILED : PAYWALLED;
}

// An engine's or an expert journey's state before setup decides it: off,
// with no accounts to run as.
function undecided() {
  return {
    enabled: false,
    code: COVERED,
    reason: "not in SCENARIOS",
    accounts: [],
  };
}

export function seedRun(auth, journeys) {
  const wants = (name) => journeys.includes(name);
  const runId = __ENV.RUN_ID || "manual";
  copilotTurnsOff();
  const wantsExperts =
    wants("expert_work") ||
    wants("expert_hire") ||
    wants("copilot_turn_scripted");
  const data = {
    auth: auth.auth,
    // The run signed the accounts up, so nothing under them predates it.
    signedUp: auth.signedUp === true,
    // The run id as it may appear in a scripted turn's nonce.
    runTag: runId.toLowerCase().replace(/[^a-z0-9]+/g, "-"),
    // Teardown deletes the chat sessions the journeys and probes created,
    // which it finds by their creation after this.
    startedAt: new Date().toISOString(),
    startsSessions: Object.keys(TURN_JOURNEYS).some(wants),
    accounts: [],
    store: discoverStore(),
    experts: wantsExperts ? discoverExperts() : noExperts(),
    graphs: { enabled: false, code: COVERED, reason: "not in SCENARIOS" },
    copilot: {
      timeouts: TURN_TIMEOUTS,
      dummy: undecided(),
      scripted: undecided(),
      real: undecided(),
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
  const turns = ["dummy", "scripted", "real"]
    .map((engine) => {
      const state = data.copilot[engine];
      return state.enabled
        ? `${engine} on for ${state.accounts.length} account(s)`
        : `${engine} off (${state.reason})`;
    })
    .join(", ");
  const experts = wantsExperts
    ? data.experts.code === COVERED
      ? `${data.experts.templates.length} templates, work expert "${data.experts.workTemplateName}" on ${
          data.accounts.filter((a) => a.experts && a.experts.workId).length
        } accounts`
      : `off (${data.experts.reason})`
    : "not in SCENARIOS";
  const line = `seed: ${data.accounts.length} accounts; subscriptions ${subscriptions}; store agents ${data.store.agents.length}, skills ${
    data.store.skillsAvailable ? data.store.skillSlugs.length : "unavailable"
  }; graph runs ${
    data.graphs.enabled
      ? `on for ${able(data, "graphRuns")} of ${data.accounts.length} accounts`
      : `off (${data.graphs.reason})`
  }; experts ${experts}; copilot turns: ${turns}`;
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
  const pendingExperts = [];
  const wantsChat = CHAT_JOURNEYS.some(wants);
  for (let i = 0; i < auth.auth.length; i++) {
    const headers = accountHeaders(auth, i);
    const account = {
      index: i,
      // The engine the account's turns run on (lib/auth.js).
      mode: auth.auth[i].mode,
      graphs: [],
      libraryAgentIds: [],
      sessionIds: [],
      // COVERED, or why this account cannot run graphs or chat: a code from
      // lib/coverage.js. `refused` is why the target refused to create a
      // graph or add a library agent for it, when it did.
      graphRuns: COVERED,
      chat: COVERED,
      refused: COVERED,
      // What the seed did about experts on this account (seedExperts), or
      // null when no expert journey runs as it.
      experts: null,
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
    if (wantsChat) {
      seedChatSessions(account, headers, data);
    }
    const hiresWork =
      (account.mode === "dummy" && wants("expert_work")) ||
      (account.mode === "scripted" && wants("copilot_turn_scripted"));
    const hires =
      hiresWork || (account.mode === "dummy" && wants("expert_hire"));
    if (hires && data.experts.code === COVERED) {
      seedExperts(account, headers, data, pendingExperts, hiresWork);
    }
  }

  if (wants("graph_run")) decideGraphRuns(data, pendingRuns);
  // The probes run while the hires install.
  decideCopilot(data, wants);
  if (pendingExperts.length > 0) waitForExperts(pendingExperts);
  for (const account of data.accounts) {
    if (account.experts && account.experts.workId) {
      seedExpertChats(account, accountHeaders(data, account.index), data);
    }
  }
  decideExperts(data, wants);
}

function able(data, field) {
  return data.accounts.filter((account) => account[field] === COVERED).length;
}

// Graphs from deterministic blocks, each also a library agent, plus one
// marketplace agent in the library when the store has one. Creating a graph
// and adding a library agent sit behind the paywall too; a 402 stops the
// account's seeding there.
function seedGraphsAndLibrary(account, headers, data, runId) {
  const number = account.index + 1;
  fundAccount(headers);
  for (let g = 0; g < SEED_GRAPHS; g++) {
    const spec = seedGraph(runId, number, g);
    const created = request("POST", "/graphs", headers, spec.body, [200, 402]);
    if (created.status === 402) {
      account.refused = refusal(created, data);
      console.warn(
        `seed: account ${number} cannot create graphs: ${detail(created)}`,
      );
      return;
    }
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
    addMarketplaceAgent(account, headers, data);
  }
  const credits = request("GET", "/credits", headers);
  if (credits.ok && credits.json)
    seedCredits.add(Number(credits.json.credits) || 0);
}

// Adds the discovered marketplace agent to the account's library. An
// account that already has it in its library keeps that entry: the seed
// neither adds it nor counts it among what teardown deletes.
function addMarketplaceAgent(account, headers, data) {
  const number = account.index + 1;
  const versionId = data.store.listingVersionId;
  const existing = request(
    "GET",
    `/library/agents/marketplace/${versionId}`,
    headers,
  );
  if (existing.ok && existing.json && existing.json.id) return;
  const added = request(
    "POST",
    "/library/agents",
    headers,
    { store_listing_version_id: versionId, source: "marketplace" },
    [200, 201, 402],
  );
  if (added.status === 402) {
    if (account.refused === COVERED) account.refused = refusal(added, data);
    console.warn(
      `seed: account ${number} cannot add a marketplace agent: ${detail(added)}`,
    );
    return;
  }
  if (added.ok && added.json && added.json.id) {
    seedLibraryAdds.add(1);
    account.libraryAgentIds.push(added.json.id);
  }
}

// SEED_RUNS runs of each of the account's graphs. A 402 keeps this account
// out of the graph journey and leaves the other accounts to try theirs.
function startSeedRuns(account, headers, data, pendingRuns) {
  const number = account.index + 1;
  if (account.graphs.length === 0) {
    if (account.refused !== COVERED) account.graphRuns = account.refused;
    else account.graphRuns = SEED_GRAPHS === 0 ? NOT_SEEDED : NO_GRAPHS;
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

// The roster's templates, as the marketplace lists them, and the one each
// dummy and scripted account hires for expert_work and the scripted expert
// chats: LOAD_TEST_EXPERT_TEMPLATE, a template's id or name, else the first.
// A target that lists none cannot serve the expert journeys, which fails a
// run that has them.
function discoverExperts() {
  const experts = noExperts();
  experts.reason = null;
  const listed = request("GET", "/experts/templates", {});
  const templates =
    listed.ok && Array.isArray(listed.json)
      ? listed.json.filter((t) => t && t.id)
      : null;
  expertTemplatesFound.add(templates ? templates.length : 0);
  if (!templates || templates.length === 0) {
    experts.code = NO_EXPERT_TEMPLATES;
    experts.reason = templates
      ? "the target lists no expert templates"
      : `listing the expert templates failed: ${detail(listed)}`;
    console.warn(`seed: ${experts.reason}; the expert journeys are off`);
    return experts;
  }
  const wanted = (__ENV.LOAD_TEST_EXPERT_TEMPLATE || "").trim();
  const work = wanted
    ? templates.find(
        (t) =>
          t.id === wanted ||
          String(t.name || "").toLowerCase() === wanted.toLowerCase(),
      )
    : templates[0];
  if (!work) {
    throw new Error(
      `LOAD_TEST_EXPERT_TEMPLATE "${wanted}" is none of the target's ${templates.length} expert templates: ${templates
        .map((t) => t.name)
        .join(", ")}`,
    );
  }
  experts.templates = templates.map((t) => t.id);
  experts.workTemplateId = work.id;
  experts.workTemplateName = work.name || work.id;
  return experts;
}

function noExperts() {
  return {
    templates: [],
    workTemplateId: null,
    workTemplateName: null,
    code: COVERED,
    reason: "not in SCENARIOS",
    work: undecided(),
    hire: undecided(),
  };
}

// What the account had before the run, and the work template's expert when
// `hiresWork`. Teardown archives every active expert of the account that is
// not among `preexisting`, which is the whole list for an account the run
// signed up; a supplied account whose experts cannot be read gets no hire,
// so nothing of its own is ever archived. expert_hire hires on the account
// only templates it has no active expert of: a re-hire would return that
// expert rather than hire.
function seedExperts(account, headers, data, pending, hiresWork) {
  const number = account.index + 1;
  const experts = {
    preexisting: null,
    workId: null,
    sessionIds: [],
    hireTemplateIds: [],
    code: COVERED,
  };
  account.experts = experts;
  if (account.chat !== COVERED) {
    experts.code = account.chat;
    return;
  }
  const listed = request("GET", "/experts", headers);
  const active = listed.ok && Array.isArray(listed.json) ? listed.json : null;
  if (active === null && !data.signedUp) {
    experts.code = NO_EXPERT;
    console.warn(
      `seed: reading account ${number}'s experts failed (${detail(listed)}); the run hires none for it`,
    );
    return;
  }
  experts.preexisting = (active || []).map((e) => e.id);
  const held = (active || []).map((e) => e.source_template_id);
  experts.hireTemplateIds = data.experts.templates.filter(
    (id) => id !== data.experts.workTemplateId && held.indexOf(id) < 0,
  );
  if (!hiresWork) return;

  const hired = request(
    "POST",
    "/experts",
    headers,
    { template_id: data.experts.workTemplateId, surface: "expert_page" },
    [200],
  );
  const expert = hired.ok && hired.json ? hired.json.expert : null;
  if (!expert || !expert.id) {
    experts.code = NO_EXPERT;
    console.warn(
      `seed: hiring the work expert for account ${number} failed: ${detail(hired)}`,
    );
    return;
  }
  experts.workId = expert.id;
  if (experts.preexisting.indexOf(expert.id) < 0) seedExpertsHired.add(1);
  if (expert.setup_status === "installing") {
    pending.push({ headers, number, id: expert.id });
  }
}

// Polls the hired experts until none is installing or the wait runs out. An
// expert whose setup failed, or is still going, still chats.
function waitForExperts(pending) {
  const deadline = Date.now() + EXPERT_SETUP_WAIT_MS;
  let open = pending.slice();
  while (open.length > 0 && Date.now() < deadline) {
    sleep(2);
    const still = [];
    for (const expert of open) {
      const res = request("GET", `/experts/${expert.id}`, expert.headers);
      const status = res.json && res.json.setup_status;
      if (!res.ok || status === "installing") {
        still.push(expert);
      } else if (status === "failed") {
        console.warn(
          `seed: account ${expert.number}'s work expert installed with failures: ${(
            res.json.setup_failures || []
          ).join(", ")}`,
        );
      }
    }
    open = still;
  }
  if (open.length > 0) {
    console.warn(
      `seed: ${open.length} work expert(s) still installing after ${EXPERT_SETUP_WAIT_MS / 1000}s; their chats open anyway`,
    );
  }
}

// The work expert's chats, named by the client as the expert page names
// them, for expert_work to open and continue.
function seedExpertChats(account, headers, data) {
  const number = account.index + 1;
  const experts = account.experts;
  for (let s = 0; s < SEED_EXPERT_SESSIONS; s++) {
    const created = request(
      "POST",
      "/chat/sessions",
      headers,
      { expert_id: experts.workId, session_id: randomUuid() },
      [200, 402],
    );
    if (!created.ok || !created.json || !created.json.id) {
      if (created.status === 402 && experts.sessionIds.length === 0) {
        experts.code = refusal(created, data);
      }
      console.warn(
        `seed: creating a chat with account ${number}'s work expert failed: ${detail(created)}`,
      );
      break;
    }
    seedExpertSessions.add(1);
    experts.sessionIds.push(created.json.id);
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

// Which accounts each expert journey runs as. expert_work needs a dummy
// account with its work expert; expert_hire a dummy account with a template
// left to hire.
function decideExperts(data, wants) {
  const dummy = data.accounts.filter((a) => a.mode === "dummy");
  const decide = (state, usable, codeFor) => {
    if (data.experts.code !== COVERED) {
      state.code = data.experts.code;
      state.reason = data.experts.reason;
      return;
    }
    if (dummy.length === 0) {
      state.code = NO_ACCOUNTS_OF_MODE;
      state.reason = "the run has no dummy account";
      return;
    }
    state.accounts = dummy.filter(usable).map((a) => a.index);
    state.enabled = state.accounts.length > 0;
    state.code = combine(dummy.map(codeFor));
    state.reason = state.code === COVERED ? null : reasonFor(state.code);
  };
  if (wants("expert_work")) {
    decide(
      data.experts.work,
      (a) => a.experts && a.experts.workId !== null,
      (a) => {
        if (a.chat !== COVERED) return a.chat;
        if (!a.experts) return NO_EXPERT;
        return a.experts.workId ? COVERED : a.experts.code || NO_EXPERT;
      },
    );
  }
  if (wants("expert_hire")) {
    decide(
      data.experts.hire,
      (a) => a.experts && a.experts.hireTemplateIds.length > 0,
      (a) => {
        if (a.chat !== COVERED) return a.chat;
        if (!a.experts || a.experts.preexisting === null) return NO_EXPERT;
        return a.experts.hireTemplateIds.length > 0
          ? COVERED
          : NO_SPARE_TEMPLATE;
      },
    );
  }
}

// What each enabled journey can exercise, from what the seed found and
// created. Returns the journeys that cannot run in full and those of them
// that failed on the target.
function recordRunCoverage(data, journeys) {
  const library = data.accounts.map(libraryCoverage);
  const chat = [combine(data.accounts.map((a) => a.chat)), able(data, "chat")];
  const engine = (state) => [
    state.code,
    state.enabled ? state.accounts.length : 0,
  ];
  // An expert journey without dummy turns still makes its reads.
  const dummyTurns = data.copilot.dummy.enabled
    ? COVERED
    : data.copilot.dummy.code;
  const expert = (state) => [
    combine([state.code, dummyTurns]),
    state.enabled ? state.accounts.length : 0,
  ];
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
    copilot_reads: chat,
    copilot_poll: chat,
    copilot_turn: engine(data.copilot.dummy),
    copilot_turn_scripted: engine(data.copilot.scripted),
    copilot_turn_real: engine(data.copilot.real),
    expert_work: expert(data.experts.work),
    expert_hire: expert(data.experts.hire),
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
  if (store.failed) return STORE_DISCOVERY_FAILED;
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
  if (account.refused !== COVERED) return account.refused;
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
// signed-in users; its 404 and an empty list are the target's limits, while
// any other answer from either list is the store failing, which fails a run
// that browses it.
function discoverStore() {
  const store = {
    agents: [],
    skillSlugs: [],
    skillsAvailable: false,
    listingVersionId: null,
    failed: false,
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
    store.failed = true;
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
  } else if (!store.failed) {
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
  } else if (skills.status === 404) {
    console.warn(
      `seed: skills are not available to visitors on the target (${detail(skills)}); the store journey skips them`,
    );
  } else {
    store.failed = true;
    console.warn(`seed: listing store skills failed: ${detail(skills)}`);
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

// Which engines the turn journeys may send turns at, and as which accounts.
// The dummy probe comes first and decides for all three: it confirms the
// copilot executor runs and the target routes accounts by address, so a
// scripted or real turn is never sent at a target that fails it. A probe
// that fails is a failure of the target, not a limit.
function decideCopilot(data, wants) {
  const wanted = Object.keys(TURN_JOURNEYS)
    .filter(wants)
    .map((name) => TURN_JOURNEYS[name]);
  if (wanted.length === 0) return;
  const engines = data.copilot;
  const decided = ["dummy"]
    .concat(wanted)
    .filter((engine, i, all) => all.indexOf(engine) === i);
  for (const engine of decided) engines[engine].reason = null;
  if (copilotTurnsOff()) {
    for (const engine of decided) {
      engines[engine].code = TURNS_OFF;
      engines[engine].reason = reasonFor(TURNS_OFF);
    }
    return;
  }

  const dummy = engines.dummy;
  const dummyAccounts = modeAccounts(data, "dummy");
  const probeAccount = dummyAccounts.find((a) => a.chat === COVERED);
  if (dummyAccounts.length === 0) {
    dummy.code = NO_ACCOUNTS_OF_MODE;
    dummy.reason = "the run has no dummy account";
  } else if (!probeAccount) {
    dummy.code = combine(dummyAccounts.map((a) => a.chat));
    dummy.reason = `no dummy account has a chat session to probe the engine with: ${reasonFor(dummy.code)}`;
  } else {
    probeDummy(data, dummy, probeAccount);
  }
  if (dummy.enabled) {
    dummy.accounts = dummyAccounts
      .filter((a) => a.chat === COVERED)
      .map((a) => a.index);
    dummy.code = combine(dummyAccounts.map((a) => a.chat));
    seedHistory(data, dummy.accounts);
  }

  for (const engine of ["scripted", "real"]) {
    if (wanted.indexOf(engine) < 0) continue;
    const state = engines[engine];
    const accounts = modeAccounts(data, engine);
    const usable = accounts.filter((a) => a.chat === COVERED);
    if (accounts.length === 0) {
      state.code = NO_ACCOUNTS_OF_MODE;
      state.reason = `the run has no ${engine} account`;
    } else if (usable.length === 0) {
      state.code = combine(accounts.map((a) => a.chat));
      state.reason = `no ${engine} account can open a chat: ${reasonFor(state.code)}`;
    } else if (!dummy.enabled) {
      // A dummy probe that failed is the reason; one that could not run
      // leaves the engine unconfirmed.
      state.code = isFailure(dummy.code) ? dummy.code : ENGINE_UNCONFIRMED;
      state.reason = isFailure(dummy.code)
        ? dummy.reason
        : reasonFor(ENGINE_UNCONFIRMED);
    } else {
      state.accounts = usable.map((a) => a.index);
      state.code = combine(accounts.map((a) => a.chat));
      state.enabled = true;
    }
  }
  if (engines.scripted.enabled) probeScripted(data, engines.scripted);
}

function modeAccounts(data, mode) {
  return data.accounts.filter((a) => a.mode === mode);
}

// One turn on a dummy account's seeded session, which must come back with
// the dummy's reply. Anything else means the account's turns would not run
// on the dummy engine, and nothing more is sent.
function probeDummy(data, dummy, account) {
  const probe = streamTurn(account.sessionIds[0], "Reply with one word.", {
    headers: accountHeaders(data, account.index),
    phase: "setup",
    timeout: PROBE_TIMEOUT,
    record: false,
    engine: "dummy",
  });
  if (probe.outcome !== "completed") {
    dummy.code = ENGINE_PROBE_FAILED;
    dummy.reason = `the dummy probe ${probe.outcome} (${probe.error || "no error"}); is the copilot executor running?`;
    return;
  }
  if (probe.text.trim() !== DUMMY_REPLY) {
    dummy.code = ENGINE_NOT_DUMMY;
    dummy.reason = reasonFor(ENGINE_NOT_DUMMY);
    return;
  }
  dummy.enabled = true;
  dummy.reason = null;
}

// One scripted turn on a new session of a scripted account, which must end
// with the script's final text. Without the scripted model, or without the
// script, the turn fails and the scripted journey is off; the others run.
function probeScripted(data, scripted) {
  const account = data.accounts[scripted.accounts[0]];
  const headers = accountHeaders(data, account.index);
  const fail = (why) => {
    scripted.enabled = false;
    scripted.accounts = [];
    scripted.code = SCRIPTED_PROBE_FAILED;
    scripted.reason = `the scripted probe ${why}: is the scripted model deployed for the target with model-scripts/loadtest-turn.json?`;
  };
  const created = request("POST", "/chat/sessions", headers, {}, [200]);
  if (!created.ok || !created.json || !created.json.id) {
    fail(`could not open its session (${detail(created)})`);
    return;
  }
  const probe = streamTurn(
    created.json.id,
    scriptedMessage(
      `probe-${data.runTag}`,
      "Check the sandbox, then answer in one sentence.",
    ),
    {
      headers,
      phase: "setup",
      timeout: data.copilot.timeouts.scripted,
      record: false,
      engine: "scripted",
    },
  );
  if (probe.outcome !== "completed" || !hasScriptedReply(probe)) {
    fail(
      probe.outcome === "completed"
        ? "completed without the script's final text"
        : `${probe.outcome} (${probe.error || "no error"})`,
    );
    return;
  }
  if (
    !probe.toolOutputs.some((output) => output.includes(SCRIPTED_TOOL_OUTPUT))
  ) {
    console.warn(
      "seed: the scripted probe finished without the sandbox command's output (held for approval, or the sandbox failed); scripted turns still run the engine",
    );
  }
}

// A little history in the dummy accounts' seeded sessions, so session reads
// carry messages. Only dummy accounts: a scripted turn needs a new session
// each time, and a real one costs money.
function seedHistory(data, indices) {
  for (const index of indices) {
    const account = data.accounts[index];
    const headers = accountHeaders(data, index);
    for (const sessionId of account.sessionIds.slice(
      0,
      SEED_TURNS_PER_ACCOUNT,
    )) {
      const turn = streamTurn(sessionId, "Say hello in one word.", {
        headers,
        phase: "setup",
        timeout: PROBE_TIMEOUT,
        record: false,
        engine: "dummy",
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
    experts_hired: count("seed_experts"),
    expert_sessions: count("seed_expert_sessions"),
    subscriptions: count("seed_subscriptions"),
    subscriptions_cancelled: count("seed_subscriptions_cancelled"),
    store_agents_found: gauge("seed_store_agents_found"),
    store_skills_found: gauge("seed_store_skills_found"),
    expert_templates_found: gauge("seed_expert_templates_found"),
    credits_per_account_avg:
      credits && credits.values ? Math.round(credits.values.avg) : null,
    executions_deleted: count("cleanup_executions"),
    library_agents_deleted: count("cleanup_library_agents"),
    sessions_deleted: count("cleanup_sessions"),
    experts_archived: count("cleanup_experts"),
    cleanup_failures: count("cleanup_failures"),
  };
}
