import type { UIMessageChunk } from "ai";
import { http, HttpResponse, type HttpHandler } from "msw";

const SSE_HEADERS = {
  "content-type": "text/event-stream",
  "cache-control": "no-cache",
  connection: "keep-alive",
  "x-vercel-ai-ui-message-stream": "v1",
  "x-accel-buffering": "no",
};

interface StreamChunksOptions {
  /** Uniform delay applied before every chunk except the first. */
  delayMsBetweenChunks?: number;
  /**
   * Per-chunk delays applied before producing each chunk. Length must match
   * `chunks`. When set, overrides `delayMsBetweenChunks`.
   */
  perChunkDelaysMs?: number[];
  /**
   * Parks the stream before a chunk until the test releases it, so a state
   * the turn only passes through stays on screen for as long as the test
   * needs to look at it, however slow the machine. See `createStreamHold`.
   */
  hold?: StreamHold;
  /**
   * Aborts in-progress chunk production. Wire this from the MSW handler's
   * `request.signal` so a consumer-side fetch abort can stop the stream and
   * unblock any pending inter-chunk delay (cancel() on the underlying
   * ReadableStream is not always invoked on abort in test environments).
   */
  abortSignal?: AbortSignal;
  /** The turn whose entries these are: every frame carries `id: <turn>:<n>-0`,
   *  as the route writes it. A fresh turn per response by default. */
  turnId?: string;
  /** Serve only the entries after this cursor, as a resume at a cursor does. */
  after?: string | null;
}

export interface StreamHold {
  /** Index of the chunk the stream stops in front of. */
  beforeChunk: number;
  released: Promise<void>;
}

/** A hold for `beforeChunk`, plus the `release` that lets the stream go on. */
export function createStreamHold(beforeChunk: number) {
  let release = () => {};
  const released = new Promise<void>((resolve) => {
    release = () => resolve();
  });
  return { hold: { beforeChunk, released } satisfies StreamHold, release };
}

let turnCounter = 0;

export function streamSseResponse(
  chunks: UIMessageChunk[],
  options: StreamChunksOptions = {},
) {
  const { delayMsBetweenChunks = 0, perChunkDelaysMs, abortSignal } = options;
  let hold = options.hold;
  const turnId = options.turnId ?? `sse-turn-${++turnCounter}`;
  const encoder = new TextEncoder();
  let nextIndex = options.after ? Number(options.after.split("-")[0]) : 0;
  let cancelled = false;
  let pendingTimer: ReturnType<typeof setTimeout> | undefined;
  let pendingResolve: (() => void) | undefined;

  function abortNow() {
    cancelled = true;
    if (pendingTimer) {
      clearTimeout(pendingTimer);
      pendingTimer = undefined;
    }
    if (pendingResolve) {
      const r = pendingResolve;
      pendingResolve = undefined;
      r();
    }
  }
  if (abortSignal) {
    if (abortSignal.aborted) cancelled = true;
    else abortSignal.addEventListener("abort", abortNow, { once: true });
  }

  function delayForIndex(i: number): number {
    if (perChunkDelaysMs) return perChunkDelaysMs[i] ?? 0;
    return i > 0 ? delayMsBetweenChunks : 0;
  }

  const stream = new ReadableStream<Uint8Array>({
    async pull(controller) {
      if (cancelled) {
        try {
          controller.close();
        } catch {
          // already closed
        }
        return;
      }
      if (hold && nextIndex === hold.beforeChunk) {
        const { released } = hold;
        hold = undefined;
        await new Promise<void>((resolve) => {
          pendingResolve = resolve;
          void released.then(resolve);
        });
        pendingResolve = undefined;
        if (cancelled) {
          try {
            controller.close();
          } catch {
            // already closed
          }
          return;
        }
      }
      const delayMs = delayForIndex(nextIndex);
      if (delayMs > 0) {
        await new Promise<void>((resolve) => {
          pendingResolve = resolve;
          pendingTimer = setTimeout(() => {
            pendingTimer = undefined;
            pendingResolve = undefined;
            resolve();
          }, delayMs);
        });
        if (cancelled) {
          try {
            controller.close();
          } catch {
            // already closed
          }
          return;
        }
      }
      if (nextIndex >= chunks.length) {
        controller.enqueue(encoder.encode("data: [DONE]\n\n"));
        controller.close();
        return;
      }
      const chunk = chunks[nextIndex++];
      controller.enqueue(
        encoder.encode(
          `id: ${turnId}:${nextIndex}-0\ndata: ${JSON.stringify(chunk)}\n\n`,
        ),
      );
    },
    cancel() {
      abortNow();
    },
  });

  return new HttpResponse(stream, { status: 200, headers: SSE_HEADERS });
}

export function assistantTextChunks(
  text: string,
  options: { messageId?: string; textPartId?: string } = {},
): UIMessageChunk[] {
  const { messageId = "test-message-1", textPartId = "test-text-1" } = options;
  return [
    { type: "start", messageId },
    { type: "start-step" },
    { type: "text-start", id: textPartId },
    { type: "text-delta", id: textPartId, delta: text },
    { type: "text-end", id: textPartId },
    { type: "finish-step" },
    { type: "finish" },
  ];
}

function streamUrl(baseUrl: string, sessionId: string) {
  return `${baseUrl}/api/chat/sessions/${sessionId}/stream`;
}

interface CopilotStreamHandlerOptions extends StreamChunksOptions {
  baseUrl: string;
  sessionId: string;
  chunks: UIMessageChunk[];
}

export function copilotStreamHandler({
  baseUrl,
  sessionId,
  chunks,
  ...streamOptions
}: CopilotStreamHandlerOptions): HttpHandler {
  return http.post(streamUrl(baseUrl, sessionId), ({ request }) =>
    streamSseResponse(chunks, {
      ...streamOptions,
      abortSignal: request.signal,
    }),
  );
}

/** A resume names its turn and cursor; one without them replays the whole turn. */
export function copilotResumeHandler({
  baseUrl,
  sessionId,
  chunks,
  ...streamOptions
}: CopilotStreamHandlerOptions): HttpHandler {
  return http.get(streamUrl(baseUrl, sessionId), ({ request }) => {
    const params = new URL(request.url).searchParams;
    return streamSseResponse(chunks, {
      turnId: params.get("turn") ?? undefined,
      after: params.get("after"),
      ...streamOptions,
      abortSignal: request.signal,
    });
  });
}

interface CopilotErrorHandlerOptions {
  baseUrl: string;
  sessionId: string;
  status: number;
  body: string | object;
}

export function copilotStreamErrorHandler({
  baseUrl,
  sessionId,
  status,
  body,
}: CopilotErrorHandlerOptions): HttpHandler {
  return http.post(streamUrl(baseUrl, sessionId), () => {
    if (typeof body === "string") {
      return new HttpResponse(body, {
        status,
        headers: { "content-type": "text/plain" },
      });
    }
    return HttpResponse.json(body, { status });
  });
}
