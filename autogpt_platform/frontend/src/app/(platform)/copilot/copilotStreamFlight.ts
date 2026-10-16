/**
 * Ms a live stream can go without receiving a byte before it is treated as
 * dead. Heartbeats count: the backend sends one every few seconds while a
 * turn is quiet (a long tool call, deep reasoning), so a healthy stream never
 * gets near this.
 */
export const STREAM_STALL_TIMEOUT_MS = 60_000;

/**
 * Bound on waiting for an aborted request to clean up before replacing it.
 * An aborted request ends within a tick (its body read and its auth wait
 * both fail with the abort), so this only guards against a hang.
 */
const SETTLE_TIMEOUT_MS = 30_000;

/** A send aborted before it reached the network is retried this many times. */
export const SEND_ATTEMPTS = 3;

/**
 * "attach" picks up a turn the chat may not be streaming yet (page load, tab
 * wake, a turn the server chained on) and leaves a healthy live stream alone.
 * "replace" swaps out a stream that was declared dead (reconnects).
 */
export type ResumeReason = "attach" | "replace";

type StartMode = ResumeReason | "wait";

/**
 * The chat's one live stream request, a send or a resume.
 *
 * The AI SDK keeps a single `activeResponse` per chat, and a request clears it
 * in its cleanup. Two overlapping requests therefore break each other: the
 * first to end clears the other's response, whose cleanup then throws
 * (Sentry BUILDER-777) and skips its `onFinish`. So a new request must wait
 * until the live one has fully ended.
 */
export interface StreamFlight {
  request: Promise<unknown> | null;
  /** The live request's ticket. */
  ticket: RequestTicket | null;
  /** Counts Stops: a start queued before one never runs. */
  stops: number;
  /** Request starts in order: each waits for the one before it to start. */
  queue: Promise<void>;
  /** A resume is queued and has not started yet. */
  resumeQueued: boolean;
  /** Counts request starts; a request carries the number it started with. */
  generation: number;
  /** When the live request last received bytes, heartbeats included. */
  lastActivityAt: number;
  abortController: AbortController | null;
}

/** One started request: whether it reached the network, and whether the
 *  flight aborted it (a replacement or Stop). */
export interface RequestTicket {
  generation: number;
  sent: boolean;
  aborted: boolean;
}

export function createStreamFlight(): StreamFlight {
  return {
    request: null,
    ticket: null,
    stops: 0,
    queue: Promise.resolve(),
    resumeQueued: false,
    generation: 0,
    lastActivityAt: 0,
    abortController: null,
  };
}

/**
 * Wrap the transport's fetch so the flight can abort any stream request and
 * sees every byte that arrives. The SDK passes no abort signal to a resume's
 * fetch, so `chat.stop()` cannot end one; this can.
 */
export function trackStreamFetch(
  inner: typeof fetch,
  flight: StreamFlight,
): typeof fetch {
  return async (input, init) => {
    // The request's own controller, made before it started, so an abort that
    // lands while the request is still preparing (auth headers) takes effect.
    const controller = flight.abortController ?? new AbortController();
    flight.abortController = controller;
    const outer = init?.signal;
    if (outer?.aborted) controller.abort(outer.reason);
    outer?.addEventListener("abort", () => controller.abort(outer.reason), {
      once: true,
    });
    flight.lastActivityAt = Date.now();
    const response = await inner(input, { ...init, signal: controller.signal });
    if (!response.body) return response;
    const body = response.body.pipeThrough(
      new TransformStream<Uint8Array, Uint8Array>({
        start(output) {
          // End the read the SDK is waiting on even if the body ignores the
          // abort, so the request always settles once it is replaced.
          const { signal } = controller;
          if (signal.aborted) output.error(signal.reason);
          else
            signal.addEventListener(
              "abort",
              () => output.error(signal.reason),
              {
                once: true,
              },
            );
        },
        transform(chunk, output) {
          flight.lastActivityAt = Date.now();
          output.enqueue(chunk);
        },
      }),
    );
    return new Response(body, {
      status: response.status,
      statusText: response.statusText,
      headers: response.headers,
    });
  };
}

/**
 * Start a send or resume once no request is live and every start queued
 * before it has run, so two can never be live at once.
 * - `replace` ends the live request first (a reconnect).
 * - `attach` stands down if a healthy request is live by its turn, and
 *   replaces a quiet one (a page-load or wake resume).
 * - `wait` waits for the live request to finish, outside the queue so a
 *   resume replacing a dead stream is never stuck behind it (a send).
 * It stands down if Stop lands after the call, or `ready` turns false; both
 * are checked before anything is aborted and again right before starting.
 * `start` gets the request's ticket, whose generation the request must carry
 * as `flightGeneration` in its SDK request metadata. Resolves, once the
 * started request has ended, with its ticket, or null if nothing started.
 */
export async function startRequest(
  flight: StreamFlight,
  start: (ticket: RequestTicket) => Promise<unknown> | undefined,
  { mode, ready = () => true }: { mode: StartMode; ready?: () => boolean },
): Promise<RequestTicket | null> {
  const stops = flight.stops;
  // `ready` first: a caller may clear its own queued flag in it.
  const canStart = () => ready() && flight.stops === stops;
  for (;;) {
    if (mode === "wait") await flight.request?.catch(() => {});
    const started = flight.queue.then(async () => {
      if (!canStart()) return null;
      if (mode === "wait" && flight.request) return "retry" as const;
      if (mode === "attach" && hasHealthyStream(flight)) return null;
      await endLiveRequest(flight);
      // Stop or an unmount can land while the live request cleans up.
      if (!canStart()) return null;
      flight.abortController = new AbortController();
      // A request is alive from its start: until its fetch runs (after the
      // auth headers) no byte can arrive, and an attach must not mistake the
      // previous request's last byte for this one going quiet.
      flight.lastActivityAt = Date.now();
      const ticket = {
        generation: ++flight.generation,
        sent: false,
        aborted: false,
      };
      flight.ticket = ticket;
      const request = start(ticket);
      if (!request) {
        flight.ticket = null;
        return null;
      }
      trackRequest(flight, request);
      return { ticket, request };
    });
    flight.queue = started.then(
      () => {},
      () => {},
    );
    const outcome = await started;
    if (outcome === "retry") continue;
    if (!outcome) return null;
    await outcome.request.catch(() => {});
    return outcome.ticket;
  }
}

interface SendHooks {
  /** Push the message and start its SDK request. */
  send: (ticket: RequestTicket) => Promise<unknown>;
  /** Take an unsent attempt's bubble back off the screen. */
  retract: () => void;
  /** Give the user their text back in the composer. */
  restoreDraft: () => void;
  /** Say the message was not sent. */
  notifyNotSent: () => void;
  isMounted: () => boolean;
}

/**
 * A user's send ends in a POST or a visible failure, never in silence. An
 * attempt aborted before it reached the network (a reconnect replaced it while
 * it waited on its auth headers) posted nothing, so it is retried; after
 * `SEND_ATTEMPTS`, a Stop, or an unmount, the user gets the text back and is
 * told it was not sent.
 */
export async function sendThroughFlight(
  flight: StreamFlight,
  hooks: SendHooks,
): Promise<"sent" | "failed" | "not-sent"> {
  const stops = flight.stops;
  const live = () => flight.stops === stops && hooks.isMounted();
  for (let attempt = 1; ; attempt++) {
    const ticket = await startRequest(flight, hooks.send, {
      mode: "wait",
      ready: live,
    });
    if (ticket?.sent) return "sent";
    if (ticket && !ticket.aborted) {
      // Failed before the network on its own (auth): the chat shows the
      // error, and the text goes back in the composer.
      hooks.restoreDraft();
      return "failed";
    }
    if (ticket) hooks.retract();
    if (!ticket || attempt >= SEND_ATTEMPTS || !live()) {
      hooks.restoreDraft();
      hooks.notifyNotSent();
      return "not-sent";
    }
  }
}

/** The user's Stop: end the live request, and let no queued start run. */
export function stopFlight(flight: StreamFlight) {
  flight.stops += 1;
  if (flight.request && flight.ticket) flight.ticket.aborted = true;
  flight.abortController?.abort();
}

function trackRequest(flight: StreamFlight, request: Promise<unknown>) {
  flight.request = request;
  void request
    .catch(() => {})
    .finally(() => {
      if (flight.request === request) flight.request = null;
    });
}

/**
 * Settle `work` with the request's own abort: a request replaced while it
 * prepares (waiting on auth headers) then ends at once, so its replacement
 * starts after it instead of beside it in the SDK.
 */
export function untilAborted<T>(flight: StreamFlight, work: Promise<T>) {
  const signal = flight.abortController?.signal;
  if (!signal) return work;
  return new Promise<T>((resolve, reject) => {
    const abort = () =>
      reject(new DOMException("Replaced before it started", "AbortError"));
    if (signal.aborted) return abort();
    signal.addEventListener("abort", abort, { once: true });
    work.then(resolve, reject).finally(() => {
      signal.removeEventListener("abort", abort);
    });
  });
}

/**
 * Called once a request's headers are ready, right before its fetch. A
 * request that was replaced while it was still preparing (the bounded wait in
 * `endLiveRequest` gave up on it) must not reach the network beside its
 * replacement; failing it as an abort ends it quietly.
 */
export function assertStillCurrent(flight: StreamFlight, metadata: unknown) {
  const generation = flightGenerationOf(metadata);
  if (generation === undefined) return;
  if (generation !== flight.generation) {
    throw new DOMException("Replaced before it started", "AbortError");
  }
  // Nothing stands between here and the fetch: the request is on the wire.
  if (flight.ticket?.generation === generation) flight.ticket.sent = true;
}

function flightGenerationOf(metadata: unknown) {
  const generation =
    typeof metadata === "object" && metadata !== null
      ? (metadata as { flightGeneration?: unknown }).flightGeneration
      : undefined;
  return typeof generation === "number" ? generation : undefined;
}

/** A request is live and has received bytes within the stall window. */
export function hasHealthyStream(flight: StreamFlight) {
  return (
    flight.request !== null &&
    Date.now() - flight.lastActivityAt < STREAM_STALL_TIMEOUT_MS
  );
}

/** Abort the live request, if any, and wait until its cleanup has run. */
export async function endLiveRequest(flight: StreamFlight) {
  const live = flight.request;
  if (!live) return;
  if (flight.ticket) flight.ticket.aborted = true;
  flight.abortController?.abort();
  let timer: ReturnType<typeof setTimeout> | undefined;
  await Promise.race([
    live.catch(() => {}),
    new Promise((resolve) => {
      timer = setTimeout(resolve, SETTLE_TIMEOUT_MS);
    }),
  ]);
  clearTimeout(timer);
}
