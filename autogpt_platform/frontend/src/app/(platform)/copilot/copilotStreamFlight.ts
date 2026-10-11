/**
 * Ms a live stream can go without receiving a byte before it is treated as
 * dead. Heartbeats count: the backend sends one every few seconds while a
 * turn is quiet (a long tool call, deep reasoning), so a healthy stream never
 * gets near this.
 */
export const STREAM_STALL_TIMEOUT_MS = 60_000;

/** Bound on waiting for an aborted request to clean up before replacing it. */
const SETTLE_TIMEOUT_MS = 5_000;

/**
 * "attach" picks up a turn the chat may not be streaming yet (page load, tab
 * wake, a turn the server chained on) and leaves a healthy live stream alone.
 * "replace" swaps out a stream that was declared dead (reconnects).
 */
export type ResumeReason = "attach" | "replace";

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
  /** Request starts in order: each waits for the one before it to start. */
  queue: Promise<void>;
  /** A resume is queued and has not started yet. */
  resumeQueued: boolean;
  /** When the live request last received bytes, heartbeats included. */
  lastActivityAt: number;
  abortController: AbortController | null;
}

export function createStreamFlight(): StreamFlight {
  return {
    request: null,
    queue: Promise.resolve(),
    resumeQueued: false,
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
    const controller = new AbortController();
    const outer = init?.signal;
    if (outer?.aborted) controller.abort(outer.reason);
    outer?.addEventListener("abort", () => controller.abort(outer.reason), {
      once: true,
    });
    flight.abortController = controller;
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
 * Start a send or resume once the live request has ended and every start
 * queued before it has run, so two can never be live at once. `start` may
 * return nothing to stand down. Resolves when the started request ends.
 */
export async function startRequest<T>(
  flight: StreamFlight,
  start: () => Promise<T> | undefined,
) {
  const started = flight.queue.then(async () => {
    await endLiveRequest(flight);
    const request = start();
    if (request) trackRequest(flight, request);
    return { request };
  });
  flight.queue = started.then(
    () => {},
    () => {},
  );
  const { request } = await started;
  return request;
}

function trackRequest<T>(flight: StreamFlight, request: Promise<T>) {
  flight.request = request;
  void request
    .catch(() => {})
    .finally(() => {
      if (flight.request === request) flight.request = null;
    });
  return request;
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
