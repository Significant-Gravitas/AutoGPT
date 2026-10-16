import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  assertStillCurrent,
  createStreamFlight,
  hasHealthyStream,
  sendThroughFlight,
  startRequest,
  stopFlight,
  untilAborted,
  type ResumeReason,
  type StreamFlight,
} from "../copilotStreamFlight";

// Seeded random interleavings of everything that starts or ends a stream
// request on one chat: user sends, attach resumes, reconnects, Stop, unmounts,
// auth lookups that stall or fail, aborts, and slow cleanup. The SDK request
// is modelled as `makeRequest` runs it through our transport: auth headers
// (raced against the request's abort), the generation check, the fetch, a
// body the abort cuts off, then cleanup. Invariants, for every seed:
// (a) every user send ends in a POST, or in a visible failure with the draft
//     given back;
// (b) after Stop, no request reaches the network until the user sends again;
// (c) at most one SDK request is alive at any time.

const SEEDS = 300;
const OPERATIONS = 40;
const HORIZON_MS = 150_000;

interface Send {
  id: number;
  outcome?: "sent" | "failed" | "not-sent";
  posted: boolean;
  restored: boolean;
  notified: boolean;
  retracted: number;
}

interface World {
  flight: StreamFlight;
  random: () => number;
  alive: number;
  maxAlive: number;
  /** `isUserStoppingRef`: cleared by a send, or by the hook once the server
   *  reports no active stream. */
  stopping: boolean;
  /** No turn to resume: the user stopped and has not sent since. */
  stoppedSinceSend: boolean;
  mounted: boolean;
  sends: Send[];
  timeline: { at: number; kind: "user" | "stop" | "network"; detail: string }[];
}

beforeEach(() => {
  vi.useFakeTimers();
});

afterEach(() => {
  vi.useRealTimers();
});

describe("stream flight — random interleavings", () => {
  it("keeps every invariant across seeded interleavings", async () => {
    const failures: string[] = [];
    const seen = { retried: 0, notSent: 0, failed: 0, sent: 0, stops: 0 };
    for (let seed = 1; seed <= SEEDS; seed++) {
      const world = await runScenario(seed);
      failures.push(...invariantViolations(seed, world));
      for (const send of world.sends) {
        if (send.retracted > 0) seen.retried += 1;
        if (send.outcome === "not-sent") seen.notSent += 1;
        if (send.outcome === "failed") seen.failed += 1;
        if (send.outcome === "sent") seen.sent += 1;
      }
      seen.stops += world.timeline.filter((e) => e.kind === "stop").length;
    }
    expect(failures.slice(0, 10)).toEqual([]);
    // The seeds reach every path the invariants are about.
    for (const [path, count] of Object.entries(seen)) {
      expect(count, path).toBeGreaterThan(0);
    }
  }, 120_000);
});

async function runScenario(seed: number): Promise<World> {
  const world: World = {
    flight: createStreamFlight(),
    random: mulberry32(seed),
    alive: 0,
    maxAlive: 0,
    stopping: false,
    stoppedSinceSend: false,
    mounted: true,
    sends: [],
    timeline: [],
  };
  const operations = Array.from({ length: OPERATIONS }, () => ({
    at: Math.floor(world.random() * HORIZON_MS),
    pick: world.random(),
  })).sort((a, b) => a.at - b.at);

  let now = 0;
  for (const operation of operations) {
    await vi.advanceTimersByTimeAsync(operation.at - now);
    now = operation.at;
    perform(world, operation.pick);
  }
  // Let every stalled lookup, body and cleanup run out.
  await vi.advanceTimersByTimeAsync(400_000);
  return world;
}

function perform(world: World, pick: number) {
  if (pick < 0.3) return userSend(world);
  if (pick < 0.5) return resume(world, "attach");
  if (pick < 0.65) return resume(world, "replace");
  if (pick < 0.78) return stop(world);
  if (pick < 0.86) {
    world.mounted = false;
    return;
  }
  if (pick < 0.94) {
    world.mounted = true;
    return;
  }
  // Nothing happens but time.
}

function userSend(world: World) {
  // `sendMessage` clears the stop flag: sending is the user acting.
  world.stopping = false;
  world.stoppedSinceSend = false;
  const send: Send = {
    id: world.sends.length + 1,
    posted: false,
    restored: false,
    notified: false,
    retracted: 0,
  };
  world.sends.push(send);
  world.timeline.push({
    at: Date.now(),
    kind: "user",
    detail: `send ${send.id}`,
  });
  void sendThroughFlight(world.flight, {
    send: (ticket) => sdkRequest(world, "POST", ticket.generation, send),
    retract: () => {
      send.retracted += 1;
    },
    restoreDraft: () => {
      send.restored = true;
    },
    notifyNotSent: () => {
      send.notified = true;
    },
    isMounted: () => world.mounted,
  }).then((outcome) => {
    send.outcome = outcome;
  });
}

// As `resumeStreamFromStart` asks for one.
function resume(world: World, reason: ResumeReason) {
  const { flight } = world;
  if (!world.mounted || world.stopping || world.stoppedSinceSend) return;
  if (flight.resumeQueued) return;
  if (reason === "attach" && hasHealthyStream(flight)) return;
  flight.resumeQueued = true;
  void startRequest(
    flight,
    (ticket) => sdkRequest(world, "GET", ticket.generation),
    {
      mode: reason,
      ready: () => {
        flight.resumeQueued = false;
        return world.mounted && !world.stopping;
      },
    },
  );
}

function stop(world: World) {
  world.stopping = true;
  world.stoppedSinceSend = true;
  world.timeline.push({ at: Date.now(), kind: "stop", detail: "stop" });
  stopFlight(world.flight);
  // Soon after, the server reports no active stream and the hook clears the
  // stop flag on its own, while starts queued before the Stop may still wait.
  if (world.random() < 0.5) {
    setTimeout(
      () => {
        world.stopping = false;
      },
      Math.floor(world.random() * 3_000),
    );
  }
}

/** `Chat.makeRequest` through the copilot transport, compressed. */
async function sdkRequest(
  world: World,
  method: "POST" | "GET",
  generation: number,
  send?: Send,
) {
  const { flight, random } = world;
  world.alive += 1;
  world.maxAlive = Math.max(world.maxAlive, world.alive);
  try {
    await untilAborted(flight, authLookup(random));
    assertStillCurrent(flight, { flightGeneration: generation });
    const signal = flight.abortController?.signal;
    world.timeline.push({
      at: Date.now(),
      kind: "network",
      detail: `${method} gen ${generation}${send ? ` send ${send.id}` : ""}`,
    });
    if (send) send.posted = true;
    await body(random, signal);
  } catch {
    // The SDK catches: an abort ends quietly, an auth failure shows an error.
  } finally {
    await sleep(Math.floor(random() * 2_000));
    world.alive -= 1;
  }
}

function authLookup(random: () => number) {
  const roll = random();
  if (roll < 0.15) return sleep(70_000); // stalls past the stall watchdog
  if (roll < 0.22) {
    return sleep(200).then(() => {
      throw new Error("Authentication failed — please sign in again.");
    });
  }
  return sleep(Math.floor(random() * 400));
}

function body(random: () => number, signal?: AbortSignal) {
  const duration = Math.floor(random() * 20_000);
  return new Promise<void>((resolve, reject) => {
    const timer = setTimeout(resolve, duration);
    signal?.addEventListener(
      "abort",
      () => {
        clearTimeout(timer);
        reject(new DOMException("aborted", "AbortError"));
      },
      { once: true },
    );
  });
}

function invariantViolations(seed: number, world: World) {
  const failures: string[] = [];
  const at = `seed ${seed}`;
  if (world.maxAlive > 1) {
    failures.push(`${at}: ${world.maxAlive} SDK requests alive at once (c)`);
  }
  for (const send of world.sends) {
    const name = `${at}: send ${send.id}`;
    if (!send.outcome) failures.push(`${name} never ended (a)`);
    if (send.outcome === "sent" && !send.posted) {
      failures.push(`${name} reported sent without a POST (a)`);
    }
    if (send.outcome === "not-sent" && (!send.restored || !send.notified)) {
      failures.push(`${name} was dropped without a visible failure (a)`);
    }
    if (send.outcome === "not-sent" && send.posted) {
      failures.push(`${name} was posted but reported not sent (a)`);
    }
    if (send.outcome === "failed" && !send.restored) {
      failures.push(`${name} failed without its draft back (a)`);
    }
  }
  let stopped = false;
  for (const event of world.timeline) {
    if (event.kind === "stop") stopped = true;
    if (event.kind === "user") stopped = false;
    if (event.kind === "network" && stopped) {
      failures.push(
        `${at}: ${event.detail} reached the network after Stop (b)`,
      );
    }
  }
  return failures;
}

function sleep(ms: number) {
  return new Promise<void>((resolve) => setTimeout(resolve, ms));
}

function mulberry32(seed: number) {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
