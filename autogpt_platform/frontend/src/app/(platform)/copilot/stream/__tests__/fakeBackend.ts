import { compareEntryIds } from "../turnConverter";
import type { PersistedRow } from "../turnLog";
import type { SessionView } from "../turnRuntime";
import type { RecordedTurn } from "./recordedTurns";

const HEARTBEAT_MS = 10_000;
const encoder = new TextEncoder();

export interface FakeConnection {
  method: string;
  url: URL;
  /** Entries already sent on this connection, by id. */
  sent: string[];
  closed: boolean;
  /** Closed by the client (an abort), as opposed to by the server. */
  aborted: boolean;
  /** End the response cleanly, as a proxy cut or the route's own end does. */
  cut(): void;
  /** Fail the response mid-body, as a dropped network does. */
  fail(): void;
  /** Stop this connection's heartbeats: it stays open and goes silent. */
  silence(): void;
}

type Responder = (request: {
  method: string;
  url: URL;
}) => "stream" | "network-error" | { status: number; body: unknown };

/**
 * The backend as the runtime sees it, for one recorded turn: the turn's
 * entries published one by one, every connection reading from its cursor,
 * heartbeats every 10 s, and a session GET whose rows follow the persists.
 */
export function fakeBackend(turn: RecordedTurn) {
  const frames = turn.sse.map((sse) => ({ id: idOf(sse), sse }));
  const turnId = turnOf(turn.sse[0]);
  let published = 0;
  let running = false;
  let started = false;
  const connections: FakeConnection[] = [];
  // Every request, refused ones included.
  const requests: { method: string; url: URL }[] = [];
  const pumps = new Set<() => void>();
  let responder: Responder = () => "stream";

  function persistedCount() {
    // A checkpoint names how many of the turn's rows are persisted.
    let rows = 0;
    for (const frame of frames.slice(0, published)) {
      const data = dataOf(frame.sse);
      if (data?.type === "data-checkpoint") rows = Number(data.data.rows);
      if (data?.type === "finish") return turn.rows.length;
    }
    return 1 + rows;
  }

  async function fetchImpl(input: RequestInfo | URL, init?: RequestInit) {
    const url = new URL(String(input));
    const method = init?.method ?? "GET";
    requests.push({ method, url });
    const answer = responder({ method, url });
    if (answer === "network-error") throw new TypeError("Failed to fetch");
    if (answer !== "stream") {
      return new Response(JSON.stringify(answer.body), {
        status: answer.status,
        headers: { "content-type": "application/json" },
      });
    }
    if (method === "POST") {
      started = true;
      running = true;
    }
    const after = url.searchParams.get("after") ?? "0-0";
    return new Response(open(method, url, after, init?.signal), {
      status: 200,
      headers: { "content-type": "text/event-stream" },
    });
  }

  function open(
    method: string,
    url: URL,
    after: string,
    signal: AbortSignal | null | undefined,
  ) {
    let index = frames.findIndex((f) => compareEntryIds(f.id, after) > 0);
    if (index === -1) index = frames.length;
    let controller!: ReadableStreamDefaultController<Uint8Array>;
    let heartbeat: ReturnType<typeof setInterval> | null = null;
    const conn: FakeConnection = {
      method,
      url,
      sent: [],
      closed: false,
      aborted: false,
      cut: () => end(() => controller.close()),
      fail: () => end(() => controller.error(new TypeError("network lost"))),
      silence: () => {
        if (heartbeat) clearInterval(heartbeat);
        heartbeat = null;
      },
    };
    function end(finish: () => void) {
      if (conn.closed) return;
      conn.closed = true;
      if (heartbeat) clearInterval(heartbeat);
      pumps.delete(pump);
      finish();
    }
    function pump() {
      while (!conn.closed && index < published) {
        const frame = frames[index++];
        conn.sent.push(frame.id);
        controller.enqueue(encoder.encode(frame.sse));
        if (dataOf(frame.sse)?.type === "finish") {
          controller.enqueue(encoder.encode("data: [DONE]\n\n"));
          end(() => controller.close());
        }
      }
    }
    connections.push(conn);
    const stream = new ReadableStream<Uint8Array>({
      start(c) {
        controller = c;
        heartbeat = setInterval(() => {
          if (!conn.closed) c.enqueue(encoder.encode(": heartbeat\n\n"));
        }, HEARTBEAT_MS);
        pumps.add(pump);
        pump();
      },
      cancel() {
        conn.aborted = true;
        end(() => {});
      },
    });
    signal?.addEventListener("abort", () => {
      conn.aborted = true;
      end(() => {});
    });
    return stream;
  }

  return {
    turnId,
    connections,
    requests,
    fetch: fetchImpl as typeof fetch,
    /** Publish the next `count` entries (default: all of them). */
    publish(count = Infinity) {
      started = true;
      running = true;
      published = Math.min(frames.length, published + count);
      if (published === frames.length) running = false;
      pumps.forEach((pump) => pump());
    },
    /** The turn is already running when the page loads. */
    beginRunning() {
      started = true;
      running = true;
    },
    respond(next: Responder) {
      responder = next;
    },
    /** The last checkpoint published, as `active_stream.checkpoint` names it. */
    lastCheckpoint() {
      for (let i = published - 1; i >= 0; i--) {
        const data = dataOf(frames[i].sse);
        if (data?.type === "data-checkpoint") {
          return {
            entry_id: frames[i].id,
            rows: Number(data.data.rows),
            sequence: Number(data.data.sequence),
          };
        }
      }
      return null;
    },
    view(rows?: PersistedRow[]): SessionView {
      const shown = started
        ? (rows ?? turn.rows.slice(0, persistedCount()))
        : [];
      return {
        messages: shown.map((row, sequence) => ({ ...row, sequence })),
        has_more_messages: false,
        active_stream: running
          ? { turn_id: turnId, checkpoint: this.lastCheckpoint() }
          : null,
      };
    },
    open() {
      return connections.filter((c) => !c.closed);
    },
  };
}

export type FakeBackend = ReturnType<typeof fakeBackend>;

function idOf(sse: string) {
  const line = sse.split("\n").find((l) => l.startsWith("id: ")) ?? "";
  return line.slice(line.lastIndexOf(":") + 1);
}

function turnOf(sse: string) {
  const line = sse.split("\n").find((l) => l.startsWith("id: ")) ?? "";
  return line.slice("id: ".length, line.lastIndexOf(":"));
}

function dataOf(sse: string) {
  const line = sse.split("\n").find((l) => l.startsWith("data: "));
  return line
    ? (JSON.parse(line.slice("data: ".length)) as {
        type: string;
        data: Record<string, unknown>;
      })
    : null;
}
