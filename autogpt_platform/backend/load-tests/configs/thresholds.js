// What a healthy target answers within, per endpoint, at the baseline
// profile: p95 in milliseconds for the steady phase. The numbers sit above
// production's p95 for each route (Sentry, autogpt-server, prod, 7 days), so
// a healthy preview passes and a regression trips them. The graph run and
// its execute route are sized on a preview instead: a whole run has no
// production number, and a preview's executor is smaller than production's.
// A profile's thresholdScale, and THRESHOLD_SCALE, multiply them.
//
// Thresholds are the run's own pass/fail. The comparison between two builds
// is perf-compare's job, in Sentry.
export const ENDPOINT_SLOS = {
  "copilot.sessions_list": { journeys: ["copilot_reads"], p95: 1500 },
  "copilot.usage": { journeys: ["copilot_reads"], p95: 800 },
  "copilot.connections": { journeys: ["copilot_reads"], p95: 2000 },
  "experts.identities": { journeys: ["copilot_reads"], p95: 800 },
  "copilot.session_detail": {
    journeys: ["copilot_reads", "copilot_turn"],
    p95: 1500,
  },
  "copilot.session_create": { journeys: ["copilot_turn"], p95: 1500 },

  "account.credits": { journeys: ["account_reads"], p95: 1500 },
  "account.subscription": { journeys: ["account_reads"], p95: 4000 },
  "account.trial": { journeys: ["account_reads"], p95: 4000 },
  "account.onboarding": { journeys: ["account_reads"], p95: 1000 },
  "account.onboarding_completed": { journeys: ["account_reads"], p95: 1000 },
  "account.credentials": { journeys: ["account_reads"], p95: 800 },
  "account.providers": { journeys: ["account_reads"], p95: 1200 },
  "account.providers_system": { journeys: ["account_reads"], p95: 300 },
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

// User-facing measurements with their own Trends.
export const CUSTOM_SLOS = {
  copilot_first_chunk_ms: { journeys: ["copilot_turn"], p95: 2000 },
  copilot_ttft_ms: { journeys: ["copilot_turn"], p95: 3000 },
  copilot_turn_ms: { journeys: ["copilot_turn"], p95: 10000 },
  // A seeded graph's run takes 26 to 34 s on a preview's executor, a few
  // seconds a node; this catches a run that stalls, not one a little slower.
  graph_run_ms: { journeys: ["graph_run"], p95: 45000 },
  graph_run_start_ms: { journeys: ["graph_run"], p95: 5000 },
  ws_connect_ms: { journeys: ["graph_run"], p95: 3000 },
  ws_event_latency_ms: { journeys: ["graph_run"], p95: 3000 },
};

export const CUSTOM_RATES = {
  copilot_turn_errors: { journeys: ["copilot_turn"], max: 0.05 },
  graph_run_errors: { journeys: ["graph_run"], max: 0.05 },
};

const MAX_FAILED_RATE = 0.02;
const MIN_CHECK_RATE = 0.98;

function enabledFor(entry, journeys) {
  return entry.journeys.some((name) => journeys.includes(name));
}

export function buildThresholds(profile, journeys) {
  const scale = profile.thresholdScale;
  const thresholds = {
    "checks{phase:steady}": [`rate>${MIN_CHECK_RATE}`],
    "http_req_failed{phase:steady}": [`rate<${MAX_FAILED_RATE}`],
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
    thresholds[`${name}{phase:steady}`] = [
      `p(95)<${Math.round(slo.p95 * scale)}`,
    ];
  }
  for (const name of Object.keys(CUSTOM_RATES)) {
    const slo = CUSTOM_RATES[name];
    if (!enabledFor(slo, journeys)) continue;
    thresholds[`${name}{phase:steady}`] = [`rate<${slo.max}`];
  }
  for (const name of journeys) {
    // An open-model scenario drops iterations when every VU is busy, which
    // means the target could not take the offered rate.
    thresholds[`dropped_iterations{scenario:${name}}`] = ["count==0"];
    thresholds[`iteration_duration{scenario:${name}}`] = [
      `p(95)<${120000 * scale}`,
    ];
  }
  return thresholds;
}

export function customMetricNames() {
  return Object.keys(CUSTOM_SLOS).concat(Object.keys(CUSTOM_RATES));
}
