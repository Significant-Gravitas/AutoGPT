/** One JSON chunk of the AI SDK UI message stream, as the backend writes it. */
export interface WireChunk {
  type: string;
  [field: string]: unknown;
}

/**
 * An `entry` is a stored stream entry with its `<turn>:<entry>` id and can be
 * resumed after. A `synthetic` frame has no id because the route wrote it
 * itself (an empty queued response or a route error). Comments are the
 * liveness signal: heartbeats and usage arrive as SSE comments.
 */
export type SseFrame =
  | { kind: "entry"; turn: string; entryId: string; chunk: WireChunk }
  | { kind: "synthetic"; chunk: WireChunk }
  | { kind: "comment"; text: string }
  | { kind: "done" };

export async function readSseFrames(
  body: ReadableStream<Uint8Array>,
  onFrame: (frame: SseFrame) => void,
  signal?: AbortSignal,
): Promise<void> {
  const parser = new SseParser(
    (id, data) => {
      const frame = toFrame(id, data);
      if (frame) onFrame(frame);
    },
    (text) => onFrame({ kind: "comment", text }),
  );
  const reader = body.getReader();
  const cancel = () => {
    reader.cancel().catch(() => {});
  };
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
    try {
      reader.releaseLock();
    } catch {
      // Already released by cancellation.
    }
  }
}

/** `<turn>:<entry>`: an entry id never holds a colon, a turn id might. */
export function parseFrameID(
  id: string,
): { turn: string; entryId: string } | null {
  const split = id.lastIndexOf(":");
  if (split <= 0 || split === id.length - 1) return null;
  return { turn: id.slice(0, split), entryId: id.slice(split + 1) };
}

function toFrame(id: string | undefined, data: string): SseFrame | null {
  if (data === "[DONE]") return { kind: "done" };
  let chunk: unknown;
  try {
    chunk = JSON.parse(data);
  } catch {
    return null;
  }
  if (
    !chunk ||
    typeof chunk !== "object" ||
    typeof (chunk as WireChunk).type !== "string"
  ) {
    return null;
  }
  const parsed = id ? parseFrameID(id) : null;
  if (!parsed) return { kind: "synthetic", chunk: chunk as WireChunk };
  return { kind: "entry", ...parsed, chunk: chunk as WireChunk };
}

/** A minimal WHATWG event-stream line parser. Ids never carry over between events. */
class SseParser {
  private buffer = "";
  private data: string[] = [];
  private id: string | undefined;

  constructor(
    private readonly onEvent: (id: string | undefined, data: string) => void,
    private readonly onComment: (text: string) => void,
  ) {}

  feed(text: string) {
    this.buffer += text;
    let start = 0;
    let index = 0;
    while (index < this.buffer.length) {
      const char = this.buffer[index];
      if (char !== "\n" && char !== "\r") {
        index += 1;
        continue;
      }
      // A trailing CR may be the first half of CRLF; wait for the next chunk.
      if (char === "\r" && index === this.buffer.length - 1) break;
      const line = this.buffer.slice(start, index);
      index += char === "\r" && this.buffer[index + 1] === "\n" ? 2 : 1;
      start = index;
      this.line(line);
    }
    this.buffer = this.buffer.slice(start);
  }

  private line(line: string) {
    if (line === "") {
      if (this.data.length > 0) this.onEvent(this.id, this.data.join("\n"));
      this.data = [];
      this.id = undefined;
      return;
    }
    if (line.startsWith(":")) {
      this.onComment(line.slice(1).trim());
      return;
    }
    const colon = line.indexOf(":");
    const field = colon === -1 ? line : line.slice(0, colon);
    let value = colon === -1 ? "" : line.slice(colon + 1);
    if (value.startsWith(" ")) value = value.slice(1);
    if (field === "data") this.data.push(value);
    else if (field === "id" && !value.includes("\u0000")) this.id = value;
  }
}
