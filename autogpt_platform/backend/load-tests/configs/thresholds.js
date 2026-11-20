// What a healthy target answers within, per endpoint, at the baseline
// profile: p95 in milliseconds for the steady phase. The numbers sit above
// production's p95 for each route (Sentry, autogpt-server, prod, 7 days), so
// a healthy preview passes and a regression trips them. The graph run, its
// execute route and its WebSocket are sized on a preview instead: a whole
// run has no production number, and a preview's executor is smaller than
// production's. A profile's thresholdScale, and THRESHOLD_SCALE, multiply
// them.
//
// Thresholds are the run's own pass/fail. The comparison between two builds
// is perf-compare's job, in Sentry.
import { durationMs, JOURNEYS, TURN_TIMEOUTS } from "./profiles.js";

export const ENDPOINT_SLOS = {
  "copilot.sessions_list": {
    journeys: ["copilot_reads", "copilot_poll", "copilot_turn", "expert_work"],
    p95: 1500,
  },
  "copilot.usage": {
    journeys: ["copilot_reads", "copilot_poll", "copilot_turn"],
    p95: 800,
  },
  "copilot.transports": {
    journeys: ["copilot_reads", "expert_hire"],
    p95: 1500,
  },
  "copilot.connections": { journeys: ["copilot_reads"], p95: 2000 },
  "copilot.session_detail": {
    journeys: [
      "copilot_reads",
      "copilot_turn",
      "copilot_turn_scripted",
      "expert_work",
      "expert_hire",
    ],
    p95: 1500,
  },
  "copilot.session_create": {
    journeys: ["copilot_turn", "copilot_turn_scripted", "copilot_turn_real"],
    p95: 1500,
  },
  "copilot.review_pending": {
    journeys: ["copilot_reads", "copilot_poll"],
    p95: 800,
  },
  "copilot.review_answered": { journeys: ["copilot_reads"], p95: 800 },
  "copilot.pending_peek": {
    journeys: ["copilot_reads", "copilot_turn"],
    p95: 800,
  },

  "experts.identities": {
    journeys: ["copilot_reads", "expert_hire"],
    p95: 800,
  },
  "experts.list": { journeys: ["expert_work", "expert_hire"], p95: 2000 },
  "experts.detail": { journeys: ["expert_work"], p95: 4000 },
  "experts.activity": { journeys: ["expert_work"], p95: 2000 },
  "experts.sessions": { journeys: ["expert_work", "expert_hire"], p95: 1500 },
  "experts.files": { journeys: ["expert_work"], p95: 1500 },
  "experts.templates": { journeys: ["expert_hire"], p95: 4000 },
  // Writes too rare in production for a p95 of their own. A hire takes the
  // account's expert lock and starts the install in the background; an
  // archive pauses the expert's schedules and stops its sandbox.
  "experts.hire": { journeys: ["expert_hire"], p95: 4000 },
  "experts.archive": { journeys: ["expert_hire"], p95: 4000 },
  "experts.session_create": {
    journeys: ["expert_work", "copilot_turn_scripted"],
    p95: 1500,
  },
  "experts.kickoff_create": { journeys: ["expert_hire"], p95: 2000 },
  "schedules.list": { journeys: ["expert_work"], p95: 2000 },
  home: { journeys: ["expert_work"], p95: 3000 },

  "account.credits": { journeys: ["account_reads"], p95: 1500 },
  "account.subscription": { journeys: ["account_reads"], p95: 4000 },
  "account.trial": { journeys: ["account_reads"], p95: 4000 },
  "account.onboarding": { journeys: ["account_reads"], p95: 1000 },
  "account.onboarding_completed": { journeys: ["account_reads"], p95: 1000 },
  "account.credentials": { journeys: ["account_reads"], p95: 800 },
  "account.providers": { journeys: ["account_reads"], p95: 1200 },
  "account.providers_system": {
    journeys: ["account_reads", "expert_hire"],
    p95: 300,
  },
  "account.orgs": { journeys: ["account_reads"], p95: 1500 },
  "account.timezone": { journeys: ["account_reads"], p95: 1500 },

  "store.skills_list": { journeys: ["store_browse"], p95: 1200 },
  "store.skill_detail": { journeys: ["store_browse"], p95: 1000 },
  "store.agents_list": { journeys: ["store_browse"], p95: 2000 },
  "store.agents_search": { journeys: ["store_browse"], p95: 2000 },
  "store.agent_detail": { journeys: ["store_browse"], p95: 800 },
  "store.creators": { journeys: ["store_browse"], p95: 1500 },
  "store.profile": { journeys: ["store_browse"], p95: 1000 },

  "library.list": { journeys: ["library_browse"], p95: 3000 },
  "library.search": { journeys: ["library_browse"], p95: 3000 },
  "library.agent_detail": { journeys: ["library_browse"], p95: 3000 },

  // Starting a run validates the graph, checks credits and queues it: about
  // 3 s on a preview, which runs with fewer resources than production.
  "graphs.execute": { journeys: ["graph_run"], p95: 4000 },
  "graphs.execution_detail": { journeys: ["graph_run"], p95: 3000 },
  "graphs.executions_list": { journeys: ["graph_run"], p95: 4000 },
  "graphs.graph_detail": { journeys: ["graph_run"], p95: 2000 },
};

// The journeys whose turns run on each copilot engine. A turn's metrics
// carry the engine as a tag (lib/copilot.js), and each engine has its own
// bounds below.
const ENGINE_JOURNEYS = {
  dummy: ["copilot_turn", "expert_work", "expert_hire"],
  scripted: ["copilot_turn_scripted"],
  real: ["copilot_turn_real"],
};

// User-facing measurements with their own Trends. A name with a tag filter,
// copilot_ttft_ms{engine:dummy}, is judged on the samples with that tag.
export const CUSTOM_SLOS = {
  // A dummy turn streams its fixed reply in seven words 0.1 s apart
  // (backend/copilot/sdk/dummy.py): a turn is the platform's own path, the
  // route, the queue, the executor and the stream relay, plus 0.7 s. No
  // production number exists for that path alone; these bound it at 2 s to
  // the first frame, 3 s to the first text and 10 s to the finish.
  "copilot_first_chunk_ms{engine:dummy,turn:chat}": {
    journeys: ENGINE_JOURNEYS.dummy,
    p95: 2000,
  },
  "copilot_ttft_ms{engine:dummy,turn:chat}": {
    journeys: ENGINE_JOURNEYS.dummy,
    p95: 3000,
  },
  "copilot_turn_ms{engine:dummy,turn:chat}": {
    journeys: ENGINE_JOURNEYS.dummy,
    p95: 10000,
  },
  // A hire's kickoff turn starts on an expert still installing, and the
  // executor holds a turn on an expert's chat until its setup is done, for
  // up to a minute (_wait_for_expert_setup in
  // backend/copilot/executor/processor.py): on a preview the wait is most of
  // the turn. The first frame comes before it; the finish within the
  // minute, plus the turn.
  "copilot_first_chunk_ms{engine:dummy,turn:kickoff}": {
    journeys: ["expert_hire"],
    p95: 2000,
  },
  "copilot_turn_ms{engine:dummy,turn:kickoff}": {
    journeys: ["expert_hire"],
    p95: 75000,
  },
  // A scripted turn runs the SDK engine whole, the sandbox, the CLI and a
  // tool round, against a model that answers at once. These are starting
  // bounds: the engine's setup time in production, with room to spare. A
  // preview's measured baseline is what to tighten them to.
  "copilot_first_chunk_ms{engine:scripted}": {
    journeys: ENGINE_JOURNEYS.scripted,
    p95: 5000,
  },
  "copilot_ttft_ms{engine:scripted}": {
    journeys: ENGINE_JOURNEYS.scripted,
    p95: 30000,
  },
  "copilot_turn_ms{engine:scripted}": {
    journeys: ENGINE_JOURNEYS.scripted,
    p95: 60000,
  },
  // A seeded graph's run takes 26 to 34 s on a preview's executor, a few
  // seconds a node; this catches a run that stalls, not one a little slower.
  graph_run_ms: { journeys: ["graph_run"], p95: 45000 },
  // Execute to the first RUNNING event: the execute call (about 3 s on a
  // preview) and the executor picking the run up. A preview whose executor
  // keeps up measured 3.4 s p50 and 4.0 s p95; 5 s leaves room for the
  // start, not for a queue.
  graph_run_start_ms: { journeys: ["graph_run"], p95: 5000 },
  // A preview opens the socket in 0.25 s p50, 0.34 s p95, and delivers an
  // event 0.9 s p50, 1.4 s p95 after its timestamp (clock skew included):
  // 3 s for either means the WebSocket server is struggling.
  ws_connect_ms: { journeys: ["graph_run"], p95: 3000 },
  ws_event_latency_ms: { journeys: ["graph_run"], p95: 3000 },
};

// Rates of failed units of work, under one in twenty: a single failed turn
// or run does not decide a baseline (thirty graph runs in its steady phase,
// about two hundred dummy turns), a pattern of them does. Scripted turns
// get one in ten: about twenty in a baseline, each with a sandbox to start.
// ws_errors counts the graph runs whose socket failed (it did not open, the
// server refused the subscription, or it dropped the connection); the
// journey runs those by polling, so graph_run_errors still measures the
// executor and ws_errors the WebSocket server.
export const CUSTOM_RATES = {
  "copilot_turn_errors{engine:dummy}": {
    journeys: ENGINE_JOURNEYS.dummy,
    max: 0.05,
  },
  "copilot_turn_errors{engine:scripted}": {
    journeys: ENGINE_JOURNEYS.scripted,
    max: 0.1,
  },
  graph_run_errors: { journeys: ["graph_run"], max: 0.05 },
  ws_errors: { journeys: ["graph_run"], max: 0.05 },
};

// Counts the summary reports for the steady phase alongside the numbers
// above. k6 keeps a tagged submetric only for a metric with a threshold, so
// each gets one that every run passes; they decide nothing. The real-model
// turns are too few for a bound, so their timings and error rate are here
// too: reported, never judged.
export const STEADY_COUNTS = {
  "copilot_first_chunk_ms{engine:real}": {
    journeys: ENGINE_JOURNEYS.real,
    always: "max>=0",
  },
  "copilot_ttft_ms{engine:real}": {
    journeys: ENGINE_JOURNEYS.real,
    always: "max>=0",
  },
  "copilot_turn_ms{engine:real}": {
    journeys: ENGINE_JOURNEYS.real,
    always: "max>=0",
  },
  "copilot_turn_errors{engine:real}": {
    journeys: ENGINE_JOURNEYS.real,
    always: "rate>=0",
  },
};
for (const engine of Object.keys(ENGINE_JOURNEYS)) {
  const journeys = ENGINE_JOURNEYS[engine];
  STEADY_COUNTS[`copilot_chunks_per_turn{engine:${engine}}`] = {
    journeys,
    always: "max>=0",
  };
  for (const name of [
    "copilot_turns",
    "copilot_turns_queued",
    "copilot_turns_failed",
  ]) {
    STEADY_COUNTS[`${name}{engine:${engine}}`] = {
      journeys,
      always: "count>=0",
    };
  }
}
Object.assign(STEADY_COUNTS, {
  graph_runs: { journeys: ["graph_run"], always: "count>=0" },
  graph_runs_failed: { journeys: ["graph_run"], always: "count>=0" },
  ws_events: { journeys: ["graph_run"], always: "count>=0" },
});

// Of all steady-phase requests, at most one in fifty may fail, and at least
// 98% of checks must pass: a healthy target fails none, so this catches a
// failing route while leaving room for a stray error from the target's own
// infrastructure.
const MAX_FAILED_RATE = 0.02;
const MIN_CHECK_RATE = 0.98;

// An iteration that takes two minutes has stalled: the longest journey, a
// graph run, takes about 35 s, and one that times out takes at most its
// socket wait, its timeout and the polling after it, about 105 s with the
// defaults (scenarios/graphs.js). A scripted turn may take its whole stream
// timeout (COPILOT_SCRIPTED_TURN_TIMEOUT) and still count, so its journey
// gets that plus half a minute for the session it creates and reads. The
// real-model turns are reported, not judged.
const MAX_ITERATION_MS = 120000;

function maxIterationMs(name) {
  if (name === "copilot_turn_scripted") {
    return Math.max(
      MAX_ITERATION_MS,
      durationMs(TURN_TIMEOUTS.scripted) + 30000,
    );
  }
  return MAX_ITERATION_MS;
}

// A custom metric's steady-phase submetric: copilot_ttft_ms{engine:dummy}
// is copilot_ttft_ms{engine:dummy,phase:steady}, graph_run_ms is
// graph_run_ms{phase:steady}.
export function steadyKey(name) {
  if (!name.endsWith("}")) return `${name}{phase:steady}`;
  return `${name.slice(0, -1)},phase:steady}`;
}

function enabledFor(entry, journeys) {
  return entry.journeys.some((name) => journeys.includes(name));
}

export function buildThresholds(profile, journeys) {
  const scale = profile.thresholdScale;
  const thresholds = {
    "checks{phase:steady}": [`rate>${MIN_CHECK_RATE}`],
    "http_req_failed{phase:steady}": [`rate<${MAX_FAILED_RATE}`],
    // setup() found a journey broken on the target (lib/coverage.js: no seed
    // run completed, an engine failed its probe, the seed could not create
    // what the journey reads) and switched it off. Its own thresholds then
    // have no samples to fail on, so this fails the run.
    coverage_failures: ["count==0"],
  };
  for (const endpoint of Object.keys(ENDPOINT_SLOS)) {
    const slo = ENDPOINT_SLOS[endpoint];
    if (!enabledFor(slo, journeys)) continue;
    thresholds[`http_req_duration{endpoint:${endpoint},phase:steady}`] = [
      `p(95)<${Math.round(slo.p95 * scale)}`,
    ];
    thresholds[`http_req_failed{endpoint:${endpoint},phase:steady}`] = [
      `rate<${MAX_FAILED_RATE}`,
    ];
  }
  for (const name of Object.keys(CUSTOM_SLOS)) {
    const slo = CUSTOM_SLOS[name];
    if (!enabledFor(slo, journeys)) continue;
    thresholds[steadyKey(name)] = [`p(95)<${Math.round(slo.p95 * scale)}`];
  }
  for (const name of Object.keys(CUSTOM_RATES)) {
    const slo = CUSTOM_RATES[name];
    if (!enabledFor(slo, journeys)) continue;
    thresholds[steadyKey(name)] = [`rate<${slo.max}`];
  }
  for (const name of Object.keys(STEADY_COUNTS)) {
    const count = STEADY_COUNTS[name];
    if (!enabledFor(count, journeys)) continue;
    thresholds[steadyKey(name)] = [count.always];
  }
  for (const name of journeys) {
    if (JOURNEYS[name].turns) {
      // Reported only. Turns the steady phase had no time for count as
      // dropped iterations.
      thresholds[`dropped_iterations{scenario:${name}}`] = ["count>=0"];
      thresholds[`iteration_duration{scenario:${name},phase:steady}`] = [
        "max>=0",
      ];
      continue;
    }
    // An open-model scenario drops iterations when every VU is busy, which
    // means the target could not take the offered rate. k6 counts a drop
    // against the scenario, without the phase, so this one covers the whole
    // run, warm-up included; a warm-up offers less than the steady rate, so
    // a drop there means the target stalled.
    thresholds[`dropped_iterations{scenario:${name}}`] = ["count==0"];
    thresholds[`iteration_duration{scenario:${name},phase:steady}`] = [
      `p(95)<${maxIterationMs(name) * scale}`,
    ];
  }
  return thresholds;
}

// Every custom metric the summary reports, each with a steady-phase
// submetric from buildThresholds (steadyKey).
export function customMetricNames() {
  return Object.keys(CUSTOM_SLOS)
    .concat(Object.keys(CUSTOM_RATES))
    .concat(Object.keys(STEADY_COUNTS));
}
