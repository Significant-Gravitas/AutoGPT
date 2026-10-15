// What a run creates under its own accounts before the load starts, so the
// journeys read and run data that exists and costs that grow with data show
// up. Everything is scoped to the run's accounts: graphs named after the
// run, runs of them, chat sessions, and one marketplace agent added to each
// library. The target's own store listings are discovered, not created.
//
// A new account on a target with credits enabled has none, and a graph run
// needs some; the one-time onboarding rewards fund each account first.
import { sleep } from "k6";
import http from "k6/http";
import { Counter, Trend } from "k6/metrics";

import { getEnvironmentConfig } from "../configs/environment.js";
import { accountCount, accountHeaders } from "./auth.js";
import { DUMMY_REPLY, streamTurn } from "./copilot.js";
import { seedGraph } from "./graphs.js";

const config = getEnvironmentConfig();
const API = config.API_BASE_URL;

const SEED_GRAPHS = whole(__ENV.SEED_GRAPHS, 4, 0, 20);
const SEED_RUNS = whole(__ENV.SEED_RUNS, 2, 0, 10);
const SEED_SESSIONS = whole(__ENV.SEED_SESSIONS, 6, 0, 50);
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

function whole(raw, fallback, min, max) {
  if (raw === undefined || raw === "") return fallback;
  const n = Number(raw);
  if (!Number.isInteger(n) || n < min || n > max) {
    throw new Error(
      `Expected a whole number from ${min} to ${max}, got "${raw}"`,
    );
  }
  return n;
}

// For options.setupTimeout: a request per seeded object, a wait for the
// seed runs, and a probe turn.
export function seedSetupTimeout() {
  const perAccount =
    2 + SEED_GRAPHS * (2 + SEED_RUNS) + SEED_SESSIONS + SEED_TURNS_PER_ACCOUNT;
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

function isPaywall(result) {
  return result.status === 402 && /subscription/i.test(detail(result));
}

export function seedRun(auth, journeys) {
  const wants = (name) => journeys.includes(name);
  const runId = __ENV.RUN_ID || "manual";
  const data = {
    auth: auth.auth,
    accounts: [],
    store: discoverStore(accountHeaders(auth, 0)),
    graphs: { enabled: wants("graph_run"), reason: null },
    copilot: {
      enabled: false,
      reason: null,
      turnTimeout: __ENV.COPILOT_TURN_TIMEOUT || "120s",
    },
  };

  const pendingRuns = [];
  for (let i = 0; i < auth.auth.length; i++) {
    const headers = accountHeaders(auth, i);
    const account = {
      index: i,
      graphs: [],
      libraryAgentIds: [],
      sessionIds: [],
    };
    data.accounts.push(account);

    if (wants("graph_run") || wants("library_browse")) {
      fundAccount(headers);
      for (let g = 0; g < SEED_GRAPHS; g++) {
        const spec = seedGraph(runId, i + 1, g);
        const created = request("POST", "/graphs", headers, spec.body);
        if (!created.ok) {
          console.warn(
            `seed: creating graph ${g + 1} for account ${i + 1} failed: ${detail(created)}`,
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

    if (wants("graph_run") && data.graphs.enabled) {
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
            data.graphs.enabled = false;
            data.graphs.reason = isPaywall(started)
              ? `the target paywalls new accounts (${detail(started)}); graph runs need an account past the paywall`
              : `the account has no credits for a run (${detail(started)})`;
            break;
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
        if (!data.graphs.enabled) break;
      }
    }

    if (wants("copilot_reads") || wants("copilot_turn")) {
      for (let s = 0; s < SEED_SESSIONS; s++) {
        const created = request(
          "POST",
          "/chat/sessions",
          headers,
          {},
          [200, 402],
        );
        if (created.status === 402) {
          data.copilot.reason = `the target paywalls new accounts (${detail(created)})`;
          console.warn(
            `seed: creating chat sessions for account ${i + 1} refused: ${detail(created)}`,
          );
          break;
        }
        if (!created.ok || !created.json || !created.json.id) {
          console.warn(
            `seed: creating a chat session for account ${i + 1} failed: ${detail(created)}`,
          );
          continue;
        }
        seedSessions.add(1);
        account.sessionIds.push(created.json.id);
      }
    }
  }

  if (wants("graph_run") && data.graphs.enabled) {
    waitForRuns(pendingRuns, data);
  }
  if (wants("copilot_turn")) {
    decideCopilot(auth, data);
  }

  console.log(
    `seed: ${data.accounts.length} accounts; store agents ${data.store.agents.length}, skills ${
      data.store.skillsAvailable ? data.store.skillSlugs.length : "unavailable"
    }; graph runs ${data.graphs.enabled ? "on" : `off (${data.graphs.reason})`}; copilot turns ${
      data.copilot.enabled
        ? "on (dummy engine confirmed)"
        : `off (${data.copilot.reason})`
    }`,
  );
  return data;
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

// The target's own marketplace: the first page of agents and skills, and
// one listing version to add to libraries.
function discoverStore(headers) {
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
  // The skills catalogue sits behind a feature flag evaluated per user.
  const skills = request(
    "GET",
    "/store/skills?page=1&page_size=50",
    headers,
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
      `seed: skills are not available on the target (${detail(skills)}); the store journey skips them`,
    );
  }
  return store;
}

// Polls the seed runs until they finish or the wait runs out. A target
// whose executor completes none of them cannot serve the graph journey.
function waitForRuns(pending, data) {
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
  seedRunsCompleted.add(completed);
  if (pending.length > 0 && completed === 0) {
    data.graphs.enabled = false;
    data.graphs.reason = `none of ${pending.length} seed runs completed within ${
      SEED_RUN_WAIT_MS / 1000
    }s (${failed} failed, ${open.length} still open): is the executor running?`;
  } else if (failed > 0 || open.length > 0) {
    console.warn(
      `seed: ${completed} of ${pending.length} seed runs completed, ${failed} failed, ${open.length} still open`,
    );
  }
}

// Copilot turns are sent only when the operator says the target runs the
// dummy engine, and after one probe turn confirms it.
function decideCopilot(auth, data) {
  const engine = __ENV.LOAD_TEST_COPILOT_ENGINE || "";
  if (data.copilot.reason) return;
  if (engine === "") {
    data.copilot.reason =
      "LOAD_TEST_COPILOT_ENGINE is not set; set it to dummy once the target runs CHAT_TEST_MODE=true on its copilot executor (README: Copilot and model spend)";
    return;
  }
  if (engine !== "dummy") {
    throw new Error(
      `LOAD_TEST_COPILOT_ENGINE=${engine}: the workload sends turns only at the dummy engine; tests/copilot-real-model.js is the opt-in real-model test`,
    );
  }
  const first = data.accounts.find((a) => a.sessionIds.length > 0);
  if (!first) {
    data.copilot.reason =
      "no chat session could be created to probe the engine";
    return;
  }
  const probe = streamTurn(first.sessionIds[0], "Reply with one word.", {
    headers: accountHeaders(auth, first.index),
    phase: "setup",
    timeout: PROBE_TIMEOUT,
    record: false,
  });
  if (probe.outcome !== "completed") {
    data.copilot.reason = `the engine probe ${probe.outcome} (${probe.error || "no error"}); is the copilot executor running?`;
    return;
  }
  if (probe.text.trim() !== DUMMY_REPLY) {
    data.copilot.reason =
      "the engine probe's reply was not the dummy's: the target is not in CHAT_TEST_MODE, and the probe spent one real turn";
    return;
  }
  data.copilot.enabled = true;

  // A little history in the seeded sessions, so session reads carry messages.
  for (const account of data.accounts) {
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

// For handleSummary: what the seed created, from k6's metrics.
export function seedingReport(summaryData) {
  const count = (name) => {
    const metric = summaryData.metrics[name];
    return metric && metric.values ? metric.values.count || 0 : 0;
  };
  const credits = summaryData.metrics.seed_credits_per_account;
  return {
    graphs: count("seed_graphs"),
    runs: count("seed_runs"),
    runs_completed: count("seed_runs_completed"),
    sessions: count("seed_sessions"),
    turns: count("seed_turns"),
    library_adds: count("seed_library_adds"),
    credits_per_account_avg:
      credits && credits.values ? Math.round(credits.values.avg) : null,
  };
}
