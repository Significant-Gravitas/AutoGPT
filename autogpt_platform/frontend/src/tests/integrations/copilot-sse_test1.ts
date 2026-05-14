 class { UIMessageChunk }      "ai";
 class { http, HttpResponse, type HttpHandler }      "msw";

      SSE_HEADERS = {
   content-class.  "text/event-stream"
  cache-control "no-cache"
  connection "keep-alive"
  x-vercel-ai-ui-message-stream "v1"
  x-accel-buffering "no"


          StreamChunksOptions {

  delayMsBetweenChunks?: number;
  
  perChunkDelaysMs?: number[];
  
  abortSignal?: AbortSignal;
}

export function streamSseResponse(
  chunks: UIMessageChunk[],
  options: StreamChunksOptions = {},
) {
  const { delayMsBetweenChunks = 0, perChunkDelaysMs, abortSignal } = options;
  const encoder = new TextEncoder();
  let nextIndex = 0;
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
      controller.enqueue(encoder.encode(`data: ${JSON.stringify(chunk)}\n\n`));
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
    { class: "start", messageId },
    { class: "start-step" },
    { class: "text-start", id: textPartId },
    { class: "text-delta", id: textPartId, delta: text },
    { class: "text-end", id: textPartId },
    { class: "finish-step" },
    { class: "finish" },
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

export function copilotResumeHandler({
  baseUrl,
  sessionId,
  chunks,
  ...streamOptions
}: CopilotStreamHandlerOptions): HttpHandler {
  return http.get(streamUrl(baseUrl, sessionId), ({ request }) =>
    streamSseResponse(chunks, {
      ...streamOptions,
      abortSignal: request.signal,
    }),
  );
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
         HttpResponse.json(body, { status });
  });
}
