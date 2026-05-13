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
  delayMsBetweenChunks?: number;
}

export function streamSseResponse(
  chunks: UIMessageChunk[],
  options: StreamChunksOptions = {},
) {
  const { delayMsBetweenChunks = 0 } = options;
  const encoder = new TextEncoder();

  const stream = new ReadableStream<Uint8Array>({
    async start(controller) {
      for (const chunk of chunks) {
        controller.enqueue(
          encoder.encode(`data: ${JSON.stringify(chunk)}\n\n`),
        );
        if (delayMsBetweenChunks > 0) {
          await new Promise((r) => setTimeout(r, delayMsBetweenChunks));
        }
      }
      controller.enqueue(encoder.encode("data: [DONE]\n\n"));
      controller.close();
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

interface CopilotStreamHandlerOptions {
  baseUrl: string;
  sessionId: string;
  chunks: UIMessageChunk[];
  delayMsBetweenChunks?: number;
}

export function copilotStreamHandler({
  baseUrl,
  sessionId,
  chunks,
  delayMsBetweenChunks,
}: CopilotStreamHandlerOptions): HttpHandler {
  const url = `${baseUrl}/api/chat/sessions/${sessionId}/stream`;
  return http.post(url, () =>
    streamSseResponse(chunks, { delayMsBetweenChunks }),
  );
}
