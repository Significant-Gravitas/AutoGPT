// The load model: which journeys run, how often, and for how long.
//
// Every journey is an open-model scenario (ramping-arrival-rate): k6 starts
// iterations at the configured rate whatever the target's latency, so a
// slower build cannot quietly send fewer requests, and queueing shows up as
// latency and as dropped_iterations rather than as a lower request count.
//
// Rates are iterations per minute at the baseline profile, weighted by the
// share of user-facing traffic each journey has in production (Sentry,
// autogpt-server, app:prod-behave:cloud). Copilot reads (session list,
// usage, session detail) are the largest family of user-facing requests,
// account reads the second, store browsing the third; library and graph runs
// are rare but slow, so they get more than their share to reach a usable
// sample size in a ten-minute run. Copilot turns are the product's core and
// its cost, so they get a sample-size floor rather than their raw share.
//
// Graph runs are also bounded by the executor. A preview's executor is at
// capacity at about ten runs a minute of the seeded graphs: at that rate a
// small slowdown queues runs without bound, and the queue slows every other
// journey with it. The baseline holds graph runs at three a minute, about
// thirty in the steady phase, so it measures a run rather than the queue;
// the stress profile's multiple of that rate approaches the capacity, to
// find where it breaks.
import exec from "k6/execution";

// Iterations per minute at the baseline profile, with the requests each
// iteration makes and the share of production traffic it stands for.
export const JOURNEYS = {
  copilot_reads: {
    rate: 60,
    exec: "copilotReads",
    requests: 5,
    why: "chat session list, usage, session detail and pending reads are the largest user-facing family (about 35% of requests)",
  },
  copilot_turn: {
    rate: 15,
    exec: "copilotTurn",
    requests: 3,
    why: "one streamed turn end to end; rare by count in production but it is the product and its cost",
  },
  account_reads: {
    rate: 30,
    exec: "accountReads",
    requests: 10,
    why: "credits, subscription, onboarding, credentials and providers: every page load makes them (about 33% of requests)",
  },
  store_browse: {
    rate: 30,
    exec: "storeBrowse",
    requests: 7,
    why: "skills list and detail, store agents and search (about 25% of requests)",
  },
  library_browse: {
    rate: 24,
    exec: "libraryBrowse",
    requests: 3,
    why: "library list and agent detail: under 1% of requests, but p50 of 11 s and 39 s in production",
  },
  graph_run: {
    rate: 3,
    exec: "graphRun",
    requests: 4,
    why: "a deterministic graph run through the executor, watched over the WebSocket: 130 to 210 runs a day in production; held well under a preview executor's capacity of about ten a minute",
  },
};

// How long each iteration of a journey tends to take, to size the VU pool.
const EXPECTED_ITERATION_SECONDS = {
  copilot_reads: 3,
  copilot_turn: 10,
  account_reads: 4,
  store_browse: 3,
  library_browse: 4,
  graph_run: 35,
};

export const PROFILES = {
  // Every journey a few times, to see that the target and the seed work.
  smoke: { scale: 0.1, warmup: "0s", duration: "1m", thresholdScale: 3 },
  // The comparison run: the rates above, for ten minutes after a warm-up.
  baseline: { scale: 1, warmup: "1m", duration: "10m", thresholdScale: 1 },
  // Three times the baseline, to find what breaks first.
  stress: { scale: 3, warmup: "1m", duration: "10m", thresholdScale: 2 },
};

export function profileName() {
  const name = (__ENV.LOAD_PROFILE || "baseline").toLowerCase();
  if (!PROFILES[name]) {
    throw new Error(
      `Unknown LOAD_PROFILE "${name}": use ${Object.keys(PROFILES).join(", ")}`,
    );
  }
  return name;
}

// The profile with the env overrides applied: RATE_SCALE multiplies every
// rate, WARMUP and DURATION replace the phase lengths.
export function loadProfile() {
  const base = PROFILES[profileName()];
  const scale = Number(__ENV.RATE_SCALE || 1);
  if (!(scale > 0)) throw new Error("RATE_SCALE must be a positive number");
  return {
    name: profileName(),
    scale: base.scale * scale,
    warmup: __ENV.WARMUP || base.warmup,
    duration: __ENV.DURATION || base.duration,
    thresholdScale: base.thresholdScale * Number(__ENV.THRESHOLD_SCALE || 1),
  };
}

// SCENARIOS=copilot_turn,store_browse runs a subset.
export function enabledJourneys() {
  const raw = __ENV.SCENARIOS;
  if (!raw) return Object.keys(JOURNEYS);
  const names = raw
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean);
  for (const name of names) {
    if (!JOURNEYS[name]) {
      throw new Error(
        `Unknown scenario "${name}" in SCENARIOS: use ${Object.keys(JOURNEYS).join(", ")}`,
      );
    }
  }
  return names;
}

export function durationMs(text) {
  const match = /^(\d+(?:\.\d+)?)(ms|s|m|h)$/.exec(String(text).trim());
  if (!match) throw new Error(`Bad duration "${text}": use 30s, 5m, 1h`);
  const unit = { ms: 1, s: 1000, m: 60000, h: 3600000 }[match[2]];
  return Number(match[1]) * unit;
}

// Iterations per minute of a journey under the profile; never below one per
// run, so smoke runs still exercise every journey.
export function journeyRate(name, profile) {
  const perMinute = JOURNEYS[name].rate * profile.scale;
  const runMinutes = durationMs(profile.duration) / 60000;
  return Math.max(perMinute, 1 / runMinutes);
}

// The k6 scenarios for the enabled journeys. Each has a warm-up stage that
// ramps to the rate and a steady stage that holds it; iterations tag their
// requests with the phase, and thresholds apply to the steady phase only.
// k6 wants whole-number rates, so they are expressed per hour.
export function buildScenarios(profile, names) {
  const scenarios = {};
  for (const name of names) {
    const journey = JOURNEYS[name];
    const rate = journeyRate(name, profile);
    const perHour = Math.max(1, Math.round(rate * 60));
    // Five times the expected concurrency, pre-allocated: k6 starts more VUs
    // up to maxVUs when it needs them, but starting one takes time and an
    // iteration due in the meantime is dropped, and a target that stalls
    // for a few seconds holds several times the usual number of VUs.
    const inFlight = (rate / 60) * EXPECTED_ITERATION_SECONDS[name];
    const preAllocatedVUs = Math.max(4, Math.ceil(inFlight * 5));
    const stages = [];
    if (durationMs(profile.warmup) > 0) {
      stages.push({ duration: profile.warmup, target: perHour });
    }
    stages.push({ duration: profile.duration, target: perHour });
    scenarios[name] = {
      executor: "ramping-arrival-rate",
      exec: journey.exec,
      startRate: stages.length > 1 ? 0 : perHour,
      timeUnit: "1h",
      stages,
      preAllocatedVUs,
      // Ten times the expected concurrency: a slow build queues inside the
      // target instead of being throttled by the client.
      maxVUs: Math.max(10, preAllocatedVUs * 10),
      gracefulStop: "90s",
      tags: { journey: name },
    };
  }
  return scenarios;
}

// Warm-up or steady, for the iteration that is starting. Every journey calls
// it first. The VU carries the phase as a tag until its next iteration, so
// k6's own per-iteration metrics, iteration_duration among them, split by
// phase as the requests do.
export function iterationPhase(profile) {
  const elapsed = Date.now() - exec.scenario.startTime;
  const phase = elapsed < durationMs(profile.warmup) ? "warmup" : "steady";
  exec.vu.metrics.tags.phase = phase;
  return phase;
}
