import { createParser } from "eventsource-parser";

import type { StreamEntry, WireChunk } from "./turnConverter";

/**
 * One SSE frame. An `entry` carries a stored entry and its id; a `synthetic`
 * data frame has no id because the route wrote it itself, so it ends this
 * connection and never the turn. Comments are surfaced because they are the
 * liveness signal: the route and the listener heartbeat through them.
 */
export type SseFrame =
  | { kind: "entry"; entry: StreamEntry }
  | { kind: "synthetic"; chunk: WireChunk }
  | { kind: "comment"; text: string }
  | { kind: "done" };

export type SseResult =
  | { ok: true }
  | { ok: false; status: number; body: unknown };

/** Open an SSE request and deliver its frames until the body ends. */
export async function fetchSse(
  url: string,
  init: RequestInit,
  onFrame: (frame: SseFrame) => void,
): Promise<SseResult> {
  const response = await fetch(url, init);
  if (!response.ok || !response.body) {
    const body = await response.json().catch(() => null);
    return { ok: false, status: response.status, body };
  }
  await readSseFrames(response.body, onFrame, init.signal ?? undefined);
  return { ok: true };
}

/** Parse an SSE byte stream into frames; resolves when the stream ends. */
export async function readSseFrames(
  body: ReadableStream<Uint8Array>,
  onFrame: (frame: SseFrame) => void,
  signal?: AbortSignal,
): Promise<void> {
  const parser = createParser({
    onEvent(event) {
      const frame = toFrame(event.id, event.data);
      if (frame) onFrame(frame);
    },
    onComment(comment) {
      onFrame({ kind: "comment", text: comment.trim() });
    },
  });
  const reader = body.getReader();
  const cancel = () => void reader.cancel().catch(() => {});
  signal?.addEventListener("abort", cancel);
  if (signal?.aborted) cancel();
  const decoder = new TextDecoder();
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      parser.feed(decoder.decode(value, { stream: true }));
    }
  } finally {
    signal?.removeEventListener("abort", cancel);
    reader.releaseLock();
  }
}

/** `<turn>:<entry>`; the entry id never holds a colon, a turn id might. */
export function parseFrameId(id: string) {
  const split = id.lastIndexOf(":");
  if (split <= 0) return null;
  return { turn: id.slice(0, split), entryId: id.slice(split + 1) };
}

function toFrame(id: string | undefined, data: string): SseFrame | null {
  if (data === "[DONE]") return { kind: "done" };
  let chunk: WireChunk;
  try {
    chunk = JSON.parse(data) as WireChunk;
  } catch {
    return null;
  }
  if (!chunk || typeof chunk.type !== "string") return null;
  const parsed = id ? parseFrameId(id) : null;
  if (!parsed) return { kind: "synthetic", chunk };
  return { kind: "entry", entry: { ...parsed, chunk } };
}
