// The load model: which journeys run, how often, and for how long.
//
// Every journey is an open-model scenario (ramping-arrival-rate): k6 starts
// iterations at the configured rate whatever the target's latency, so a
// slower build cannot quietly send fewer requests, and queueing shows up as
// latency and as dropped_iterations rather than as a lower request count.
// The real-model turns are the exception: a fixed, capped number of them,
// one at a time, because each one costs money.
//
// Rates are iterations per minute at the baseline profile, weighted by the
// share of user-facing traffic each family has in production (Sentry,
// autogpt-server, app:prod-behave:cloud, seven days): copilot about 60% of
// requests, store about 17%, account reads about 16%, experts about 2%,
// graphs and the library about 1%. Copilot and experts are the product's
// primary interaction points. The copilot journeys (opening a chat, the open
// chat's polling, turns on three engines) make about three-fifths of the
// run's requests, as in production, and together with the expert journeys
// about two-thirds. Experts get well above their raw share because users
// work with them far more often than they hire them: the expert work
// journey runs six times as often as the hire. Store and account reads
// follow their production shares. Library browsing, graph runs, scripted
// turns and hires are rare but slow, so they get a sample-size floor in a
// ten-minute run rather than their raw share.
//
// The whole workload is bounded by a preview's API server, which has one
// CPU. At about 1,070 requests a minute of this mix it saturates: its own
// Redis calls time out, it falls back to reading the database for
// everything, its transactions expire, and it runs out of memory within
// minutes. The baseline sends about 670 a minute, so it measures latency
// below that capacity, not the queue past it; the stress profile, three
// times the baseline, finds where it breaks.
//
// Graph runs are also bounded by the executor. A preview's executor is at
// capacity at about ten runs a minute of the seeded graphs: at that rate a
// small slowdown queues runs without bound, and the queue slows every other
// journey with it. The baseline holds graph runs at three a minute, about
// thirty in the steady phase, so it measures a run rather than the queue;
// the stress profile's multiple of that rate approaches the capacity, to
// find where it breaks.
import exec from "k6/execution";

// The most real-model turns a run may send (LOAD_TEST_COPILOT_REAL_TURNS).
export const MAX_REAL_TURNS = 10;

// How many turns copilot_turn_real sends: LOAD_TEST_COPILOT_REAL_TURNS, 0
// (the default) to send none. Usable in the init context.
export function realTurns() {
  const raw = __ENV.LOAD_TEST_COPILOT_REAL_TURNS;
  if (raw === undefined || raw === "") return 0;
  const n = Number(raw);
  if (!Number.isInteger(n) || n < 0 || n > MAX_REAL_TURNS) {
    throw new Error(
      `LOAD_TEST_COPILOT_REAL_TURNS must be a whole number from 0 to ${MAX_REAL_TURNS}, got "${raw}"`,
    );
  }
  return n;
}

// Iterations per minute at the baseline profile, with the requests each
// iteration makes and the share of production traffic it stands for. A
// journey with `turns` instead of a rate runs that many iterations once.
export const JOURNEYS = {
  copilot_reads: {
    rate: 18,
    exec: "copilotReads",
    requests: 9,
    why: "opening a chat: the session list (48k requests a week in production), usage (17k), transports, connections and experts, then one session's detail, review cards (75k, the largest single route) and queued messages",
  },
  copilot_poll: {
    rate: 24,
    exec: "copilotPoll",
    requests: 6.3,
    why: "an open chat waiting on a review card: the page polls its cards every 2 s, the session list every 10 s and usage every 30 s, which makes the review route the largest in production (75k requests a week, p95 227 ms)",
  },
  copilot_turn: {
    rate: 9,
    exec: "copilotTurn",
    requests: 6.1,
    why: "a streamed turn on the dummy engine, the platform's own path without a model, with the reads the page makes after it: rare by count in production (about 850 streams a week) but it is the product",
  },
  copilot_turn_scripted: {
    rate: 2,
    exec: "copilotTurnScripted",
    requests: 3,
    why: "a turn on the whole SDK engine (sandbox, CLI, a tool round) against the scripted model, at no model cost; a sample-size floor, since each one starts a sandbox",
  },
  copilot_turn_real: {
    turns: realTurns,
    exec: "copilotTurnReal",
    requests: 2,
    why: "a few capped, opt-in turns on the real model, for what users feel with a model behind the chat; a one-step turn costs about $0.07 in production",
  },
  expert_work: {
    rate: 4,
    exec: "expertWork",
    requests: 12,
    why: "working with a hired expert: its page (the expert, its activity, schedules, home, its chats and files) and a turn in one of its chats; well above the experts' 2% of requests, since users work with experts far more often than they hire them",
  },
  expert_hire: {
    rate: 1,
    exec: "expertHire",
    requests: 16,
    why: "hiring an expert from the marketplace and its kickoff turn, then the team page polling while it installs; rare (about 900 template reads a week in production) but slow, p50 1.3 s for the templates and 3.0 s p95 for an expert's page",
  },
  account_reads: {
    rate: 10,
    exec: "accountReads",
    requests: 10,
    why: "credits, subscription, onboarding, credentials and providers: every page load makes them (about 16% of requests)",
  },
  store_browse: {
    rate: 15,
    exec: "storeBrowse",
    requests: 7,
    why: "skills list and detail, store agents and search (about 17% of requests)",
  },
  library_browse: {
    rate: 4,
    exec: "libraryBrowse",
    requests: 3,
    why: "library list and agent detail: about 1% of requests with graphs, but p50 of 11 s and 39 s in production",
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
  copilot_reads: 4,
  copilot_poll: 10,
  copilot_turn: 10,
  copilot_turn_scripted: 60,
  expert_work: 10,
  expert_hire: 30,
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

// The cap on one turn's stream, per engine. A dummy turn takes about a
// second; a scripted turn starts a sandbox and the CLI before its first
// word; a real turn waits on a model that may call tools for minutes.
export const TURN_TIMEOUTS = {
  dummy: __ENV.COPILOT_TURN_TIMEOUT || "120s",
  scripted: __ENV.COPILOT_SCRIPTED_TURN_TIMEOUT || "180s",
  real: "300s",
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
// rate, THRESHOLD_SCALE every threshold, WARMUP and DURATION replace the
// phase lengths.
export function loadProfile() {
  const base = PROFILES[profileName()];
  return {
    name: profileName(),
    scale: base.scale * positiveNumber("RATE_SCALE"),
    warmup: __ENV.WARMUP || base.warmup,
    duration: __ENV.DURATION || base.duration,
    thresholdScale: base.thresholdScale * positiveNumber("THRESHOLD_SCALE"),
  };
}

// A multiplier from the environment, 1 when unset. Anything that is not a
// finite positive number stops the run at init rather than turning every
// rate or threshold into NaN.
function positiveNumber(name) {
  const raw = __ENV[name];
  if (raw === undefined || raw === "") return 1;
  const value = Number(raw);
  if (!(Number.isFinite(value) && value > 0)) {
    throw new Error(`${name} must be a positive number, got "${raw}"`);
  }
  return value;
}

// SCENARIOS=copilot_turn,store_browse runs a subset. copilot_turn_real runs
// only when LOAD_TEST_COPILOT_REAL_TURNS asks for turns.
export function enabledJourneys() {
  const raw = __ENV.SCENARIOS;
  if (!raw) {
    return Object.keys(JOURNEYS).filter(
      (name) => !JOURNEYS[name].turns || JOURNEYS[name].turns() > 0,
    );
  }
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
    if (JOURNEYS[name].turns && JOURNEYS[name].turns() === 0) {
      throw new Error(
        `SCENARIOS names ${name}, which needs LOAD_TEST_COPILOT_REAL_TURNS from 1 to ${MAX_REAL_TURNS}`,
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
// run, so smoke runs still exercise every journey. A journey with a fixed
// number of turns has no rate.
export function journeyRate(name, profile) {
  if (JOURNEYS[name].turns) return null;
  const perMinute = JOURNEYS[name].rate * profile.scale;
  const runMinutes = durationMs(profile.duration) / 60000;
  return Math.max(perMinute, 1 / runMinutes);
}

// What a journey offers the target under the profile, for the log.
export function journeyLoad(name, profile) {
  const journey = JOURNEYS[name];
  if (journey.turns) return `${journey.turns()} turns`;
  return `${journeyRate(name, profile).toFixed(1)}/min`;
}

// The k6 scenarios for the enabled journeys. Each has a warm-up stage that
// ramps to the rate and a steady stage that holds it; iterations tag their
// requests with the phase, and thresholds apply to the steady phase only.
// k6 wants whole-number rates, so they are expressed per hour.
export function buildScenarios(profile, names) {
  const scenarios = {};
  for (const name of names) {
    const journey = JOURNEYS[name];
    if (journey.turns) {
      // One VU sends the turns back to back through the steady phase, and
      // whatever it has not sent when the phase ends stays unsent.
      scenarios[name] = {
        executor: "shared-iterations",
        exec: journey.exec,
        vus: 1,
        iterations: journey.turns(),
        startTime: profile.warmup,
        maxDuration: profile.duration,
        gracefulStop: "90s",
        tags: { journey: name },
      };
      continue;
    }
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
// phase as the requests do. `startOffsetMs` is how long after the load began
// the journey's scenario starts: the real-model turns start after the
// warm-up, so all of them are steady.
export function iterationPhase(profile, startOffsetMs) {
  const elapsed =
    Date.now() - exec.scenario.startTime + (Number(startOffsetMs) || 0);
  const phase = elapsed < durationMs(profile.warmup) ? "warmup" : "steady";
  exec.vu.metrics.tags.phase = phase;
  return phase;
}
