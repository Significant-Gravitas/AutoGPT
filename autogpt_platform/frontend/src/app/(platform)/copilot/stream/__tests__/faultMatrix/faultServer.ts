import { compareEntryIds } from "../../turnConverter";
import type { PersistedRow } from "../../turnLog";
import type { SessionView } from "../../turnRuntime";
import { loadRecordedTurn } from "../recordedTurns";

const FIXTURE = loadRecordedTurn("baseline-tool-turn");
const HEARTBEAT_MS = 10_000;
const LISTENER_READ_MS = 5_000;
const encoder = new TextEncoder();
const QUEUED_STREAM = emptyStream();
const FOLLOW_UP_STREAM = emptyStream({
  type: "data-follow-up-queued",
  data: {},
  transient: true,
});

// The session GET's rows carry the message id the renderer keys a prompt on.
type Row = PersistedRow & { id: string };

/** How the next send is answered, or lost, between the client and the server. */
export type PostMode =
  | "stream"
  | "lost-before-send"
  | "lost-after-receive"
  // The server started the turn; the response stays open and silent, as a frozen page sees it.
  | "silent-after-receive"
  | "route-error-after-receive"
  | { status: number; body: unknown };

export type ResumeMode =
  | "stream"
  | "network-error"
  | { status: number; body: unknown };

export interface FaultConnection {
  method: string;
  url: URL;
  closed: boolean;
  aborted: boolean;
  /** Fail the response mid-body, as a dropped socket does. */
  fail(): void;
}

interface FaultTurn {
  id: string;
  frames: { id: string; sse: string }[];
  /** The turn's own rows after its prompt, with absolute sequences. */
  rows: Row[];
  published: number;
  state: "queued" | "running" | "ended" | "stopped";
}

/**
 * The chat backend as the turn runtime meets it, over several turns: a send
 * persists its prompt and registers its turn before the response starts, as
 * `stream_chat_post` does; a send while a turn runs joins its pending queue
 * and becomes the next turn; a resume serves a turn's entries after a cursor.
 * Every turn replays the recorded `baseline-tool-turn` under its own id.
 */
export class FaultServer {
  history: Row[] = [];
  turns: FaultTurn[] = [];
  connections: FaultConnection[] = [];
  requests: { method: string; url: URL }[] = [];
  postMode: PostMode | (() => PostMode) = "stream";
  resumeMode: ResumeMode = "stream";
  /** Session reads fail while this is above zero; each read takes one. */
  failViews = 0;
  /** Session reads fail while this holds, as for a signed-out user. */
  viewsDown = false;
  /** The user is at the running cap: the next send is queued. */
  queueing = false;
  viewReads = 0;
  private pending: string[] = [];
  private pumps = new Set<() => void>();

  fetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = new URL(String(input));
    const method = init?.method ?? "GET";
    this.requests.push({ method, url });
    return method === "POST" ? this.post(url, init) : this.resume(url, init);
  }) as typeof fetch;

  view(): SessionView {
    this.viewReads += 1;
    if (this.viewsDown) throw new TypeError("Failed to fetch");
    if (this.failViews > 0) {
      this.failViews -= 1;
      throw new TypeError("Failed to fetch");
    }
    const turn = this.current();
    return {
      messages: this.history.map((row) => ({ ...row })),
      has_more_messages: false,
      active_stream:
        turn?.state === "running"
          ? { turn_id: turn.id, checkpoint: this.checkpoint(turn) }
          : null,
      chat_status:
        turn?.state === "queued"
          ? "queued"
          : turn?.state === "running"
            ? "running"
            : "idle",
    };
  }

  /** Persist a prompt and register its turn, as `schedule_chat_turn` does. */
  startTurn(text: string, state: FaultTurn["state"] = "running") {
    const id = `turn-${this.turns.length + 1}`;
    const promptSequence = this.history.length;
    const turn: FaultTurn = {
      id,
      frames: FIXTURE.sse.map((sse, i) => ({
        id: `${i + 1}-0`,
        sse: sse
          .replace("id: drift-turn:", `id: ${id}:`)
          .replace('"sequence": 1,', `"sequence": ${promptSequence + 1},`),
      })),
      rows: FIXTURE.rows.slice(1).map((row, i) => ({
        ...row,
        id: `row-${promptSequence + 1 + i}`,
        sequence: promptSequence + 1 + i,
      })),
      published: 0,
      state,
    };
    this.turns.push(turn);
    this.history.push({
      ...FIXTURE.rows[0],
      id: `row-${promptSequence}`,
      content: text,
      sequence: promptSequence,
    });
    // The route publishes "Message received…" before it answers.
    if (state === "running") this.publish(1);
    return turn;
  }

  /** Publish the running turn's next `count` entries (default: to its end). */
  publish(count = Infinity) {
    const turn = this.running();
    if (!turn) return;
    const before = persistedRows(turn);
    turn.published = Math.min(turn.frames.length, turn.published + count);
    this.history.push(...turn.rows.slice(before, persistedRows(turn)));
    if (turn.published === turn.frames.length) {
      turn.state = "ended";
      if (this.pending.length)
        this.startTurn(this.pending.splice(0).join("\n"));
    }
    this.pumps.forEach((pump) => pump());
  }

  /** The queued turn gets its slot. */
  startQueued() {
    const turn = this.current();
    if (turn?.state !== "queued") return;
    turn.state = "running";
    this.publish(1);
  }

  /** The turn is stopped elsewhere; no finish is stored. */
  stopWithoutFinish() {
    const turn = this.running();
    if (!turn) return;
    turn.state = "stopped";
    this.pumps.forEach((pump) => pump());
  }

  private async post(url: URL, init?: RequestInit) {
    const mode =
      typeof this.postMode === "function" ? this.postMode() : this.postMode;
    if (mode === "lost-before-send") throw new TypeError("Failed to fetch");
    if (typeof mode === "object") return refusal(mode.status, mode.body);
    const { message } = JSON.parse(String(init?.body)) as { message: string };
    if (this.running()) {
      this.pending.push(message);
      return sse(FOLLOW_UP_STREAM);
    }
    const turn = this.startTurn(message, this.queueing ? "queued" : "running");
    if (turn.state === "queued") return sse(QUEUED_STREAM);
    if (mode === "lost-after-receive") throw new TypeError("Failed to fetch");
    if (mode === "route-error-after-receive")
      return sse(
        [
          { type: "error", errorText: "An error occurred. Please try again." },
          { type: "finish" },
        ]
          .map((chunk) => `data: ${JSON.stringify(chunk)}\n\n`)
          .concat("data: [DONE]\n\n")
          .join(""),
      );
    return sse(
      this.open(
        "POST",
        url,
        turn,
        "0-0",
        init?.signal,
        mode === "silent-after-receive",
      ),
    );
  }

  private async resume(url: URL, init?: RequestInit) {
    const mode = this.resumeMode;
    if (mode === "network-error") throw new TypeError("Failed to fetch");
    if (typeof mode === "object") return refusal(mode.status, mode.body);
    const turn = this.turns.find((t) => t.id === url.searchParams.get("turn"));
    if (!turn) return refusal(410, { reason: "expired" });
    const after = url.searchParams.get("after") ?? "0-0";
    return sse(this.open("GET", url, turn, after, init?.signal));
  }

  private open(
    method: string,
    url: URL,
    turn: FaultTurn,
    after: string,
    signal: AbortSignal | null | undefined,
    silent = false,
  ) {
    let index = turn.frames.findIndex((f) => compareEntryIds(f.id, after) > 0);
    if (index === -1) index = turn.frames.length;
    let controller!: ReadableStreamDefaultController<Uint8Array>;
    let heartbeat: ReturnType<typeof setInterval> | null = null;
    let done: ReturnType<typeof setTimeout> | null = null;
    const pumps = this.pumps;
    const conn: FaultConnection = {
      method,
      url,
      closed: false,
      aborted: false,
      fail: () => end(() => controller.error(new TypeError("network lost"))),
    };
    function end(finish: () => void) {
      if (conn.closed) return;
      conn.closed = true;
      if (heartbeat) clearInterval(heartbeat);
      if (done) clearTimeout(done);
      pumps.delete(pump);
      finish();
    }
    function pump() {
      if (silent) return;
      while (!conn.closed && index < turn.published) {
        const frame = turn.frames[index++];
        controller.enqueue(encoder.encode(frame.sse));
        if (isFinish(frame.sse)) {
          controller.enqueue(encoder.encode("data: [DONE]\n\n"));
          end(() => controller.close());
        }
      }
      // The listener ends a read of a turn that is no longer running.
      if (!conn.closed && !done && turn.state !== "running") {
        done = setTimeout(() => {
          controller.enqueue(encoder.encode("data: [DONE]\n\n"));
          end(() => controller.close());
        }, LISTENER_READ_MS);
      }
    }
    this.connections.push(conn);
    signal?.addEventListener("abort", () => {
      conn.aborted = true;
      end(() => {});
    });
    return new ReadableStream<Uint8Array>({
      start(c) {
        controller = c;
        if (!silent)
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
  }

  private current(): FaultTurn | null {
    return this.turns[this.turns.length - 1] ?? null;
  }

  private running(): FaultTurn | null {
    const turn = this.current();
    return turn?.state === "running" ? turn : null;
  }

  private checkpoint(turn: FaultTurn) {
    for (let i = turn.published - 1; i >= 0; i--) {
      const data = dataOf(turn.frames[i].sse);
      if (data?.type === "data-checkpoint") {
        return {
          entry_id: turn.frames[i].id,
          rows: Number(data.data.rows),
          sequence: Number(data.data.sequence),
        };
      }
    }
    return null;
  }
}

// A checkpoint names how many of the turn's rows are persisted; the finish, all of them.
function persistedRows(turn: FaultTurn) {
  let rows = 0;
  for (const frame of turn.frames.slice(0, turn.published)) {
    if (isFinish(frame.sse)) return turn.rows.length;
    const data = dataOf(frame.sse);
    if (data?.type === "data-checkpoint") rows = Number(data.data.rows);
  }
  return rows;
}

function isFinish(sse: string) {
  return dataOf(sse)?.type === "finish";
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

function emptyStream(...parts: object[]) {
  return [
    { type: "start", messageId: "empty" },
    { type: "start-step" },
    ...parts,
    { type: "finish-step" },
    { type: "finish" },
  ]
    .map((chunk) => `data: ${JSON.stringify(chunk)}\n\n`)
    .concat("data: [DONE]\n\n")
    .join("");
}

function sse(body: string | ReadableStream<Uint8Array>) {
  return new Response(body, {
    status: 200,
    headers: { "content-type": "text/event-stream" },
  });
}

function refusal(status: number, body: unknown) {
  const html = typeof body === "string";
  return new Response(html ? body : JSON.stringify(body), {
    status,
    headers: { "content-type": html ? "text/html" : "application/json" },
  });
}
