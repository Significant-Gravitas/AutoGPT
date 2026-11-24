import { compareEntryIds } from "../../turnConverter";
import type { PersistedRow } from "../../turnLog";
import type { SessionView } from "../../turnRuntime";
import { persistedPromptId } from "../../turnTail";
import { loadRecordedTurn } from "../recordedTurns";

export const USER_ID = "user-1";
const FIXTURE = loadRecordedTurn("baseline-tool-turn");
const HEARTBEAT_MS = 10_000;
// Past the client's silence abort (30 s) and its checks (4 s).
const LATE_RECEIVE_MS = 40_000;
const LISTENER_READ_MS = 5_000;
const encoder = new TextEncoder();
const QUEUED_STREAM = emptyStream();
const FOLLOW_UP_STREAM = emptyStream({
  type: "data-follow-up-queued",
  data: {},
  transient: true,
});
const FOLLOW_UP_UNFINISHED = FOLLOW_UP_STREAM.replace("data: [DONE]\n\n", "");

// The session GET's rows carry the message id the renderer keys a prompt on.
type Row = PersistedRow & { id: string };

interface PostedBody {
  message: string;
  message_id: string;
  file_ids: string[] | null;
  folder_ids: string[] | null;
}

/** How the next send is answered, or lost, between the client and the server. */
export type PostMode =
  | "stream"
  | "lost-before-send"
  // The request never reaches the server and never answers, as a dead socket holds it.
  | "hang-before-send"
  | "lost-after-receive"
  // The request reaches the server only after the client gave up on it.
  | "late-after-receive"
  // The client's request fails at once, but the server takes it that late too.
  | "lost-late-receive"
  // Sent while a turn runs: the server queues it and says so, then the
  // response fails, as a dropped socket does.
  | "lost-after-follow-up"
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
  /** Each POST's body, in order. */
  posts: PostedBody[] = [];
  /** The next POST's answer reaches the client this much later; the server
   *  has the message at once, as over a slow link. */
  respondAfterMs = 0;
  postMode: PostMode | (() => PostMode) = "stream";
  resumeMode: ResumeMode = "stream";
  /** Session reads fail while this is above zero; each read takes one. */
  failViews = 0;
  /** Session reads fail while this holds, as for a signed-out user. */
  viewsDown = false;
  /** The user is at the running cap: the next send is queued. */
  queueing = false;
  /** The session GET's window: only the latest this-many rows. */
  window: number | null = null;
  /** The next session GET lists this many of the newest rows late, as one
   *  that reads the messages just before a send's row and turn land. */
  lagRowsOnce = 0;
  /** Views name the session's owner, as a read the page made after a send does. */
  namesUser = true;
  /** Runs once as the next session GET starts, as a page read landing meanwhile. */
  beforeViewOnce: (() => void) | null = null;
  /** A response stays open this much longer after its turn's finish, as a
   *  proxy can hold it. */
  closeAfterFinishMs = 0;
  viewReads = 0;
  private pending: { text: string; id: string }[] = [];
  private pumps = new Set<() => void>();

  fetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = new URL(String(input));
    const method = init?.method ?? "GET";
    this.requests.push({ method, url });
    if (method !== "POST") return this.resume(url, init);
    this.posts.push(JSON.parse(String(init?.body)) as PostedBody);
    const delay = this.respondAfterMs;
    this.respondAfterMs = 0;
    const response = await this.post(url, init);
    if (delay) await new Promise((resolve) => setTimeout(resolve, delay));
    return response;
  }) as typeof fetch;

  view(): SessionView {
    const before = this.beforeViewOnce;
    this.beforeViewOnce = null;
    before?.();
    this.viewReads += 1;
    if (this.viewsDown) throw new TypeError("Failed to fetch");
    if (this.failViews > 0) {
      this.failViews -= 1;
      throw new TypeError("Failed to fetch");
    }
    const turn = this.current();
    const lag = this.lagRowsOnce;
    this.lagRowsOnce = 0;
    const read = lag ? this.history.slice(0, -lag) : this.history;
    const rows = this.window === null ? read : read.slice(-this.window);
    return {
      user_id: this.namesUser ? USER_ID : null,
      messages: rows.map((row) => ({ ...row })),
      has_more_messages: rows.length < read.length,
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

  /**
   * Persist a prompt and register its turn, as `schedule_chat_turn` does. A
   * send's prompt row takes the id the backend scopes from its `message_id`.
   */
  startTurn(
    text: string,
    state: FaultTurn["state"] = "running",
    promptId: string | null = null,
  ) {
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
      id: promptId ?? `row-${promptSequence}`,
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
      // `persist_pending_as_user_rows`: each queued send its own row, then the turn.
      const drained = this.pending.splice(0);
      const last = drained.pop();
      for (const queued of drained) {
        const sequence = this.history.length;
        this.history.push({
          ...FIXTURE.rows[0],
          id: queued.id,
          content: queued.text,
          sequence,
        });
      }
      if (last) this.startTurn(last.text, "running", last.id);
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
    if (mode === "hang-before-send") return hangUntilAborted(init?.signal);
    if (typeof mode === "object") return refusal(mode.status, mode.body);
    const body = JSON.parse(String(init?.body)) as {
      message: string;
      message_id: string;
      expert_kickoff: boolean;
    };
    // `expert_kickoff_message_id`: a kickoff's row is keyed on its expert.
    const id = body.expert_kickoff
      ? "expert-kickoff-row"
      : scopedMessageId(url, body.message_id);
    if (mode === "late-after-receive" || mode === "lost-late-receive") {
      setTimeout(() => this.arrive(body.message, id), LATE_RECEIVE_MS);
      if (mode === "lost-late-receive") throw new TypeError("Failed to fetch");
      return hangUntilAborted(init?.signal);
    }
    // The route's claim on the id: a send it already took starts nothing, and
    // is handed the turn running now, if any.
    if (this.took(id)) {
      const running = this.running();
      return sse(
        running
          ? this.open("POST", url, running, "0-0", init?.signal)
          : emptyStream(),
      );
    }
    if (this.running()) {
      this.pending.push({ text: body.message, id });
      if (mode === "lost-after-receive") throw new TypeError("Failed to fetch");
      if (mode === "lost-after-follow-up")
        return sse(failsAfter(FOLLOW_UP_UNFINISHED));
      return sse(FOLLOW_UP_STREAM);
    }
    // The route matches a send on its text too only when it names no id: one
    // that repeats the trailing user rows is dropped, and no turn starts.
    if (!body.message_id && this.repeatsTrailingPrompt(body.message))
      return sse(emptyStream());
    const turn = this.startTurn(
      body.message,
      this.queueing ? "queued" : "running",
      id,
    );
    if (mode === "lost-after-receive") throw new TypeError("Failed to fetch");
    if (turn.state === "queued") return sse(QUEUED_STREAM);
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
    const closeAfterFinishMs = this.closeAfterFinishMs;
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
          if (!closeAfterFinishMs) end(() => controller.close());
          else
            done = setTimeout(
              () => end(() => controller.close()),
              closeAfterFinishMs,
            );
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

  // The route's claim on a send's id: taken as a row, or queued.
  private took(id: string) {
    return (
      this.history.some((row) => row.id === id) ||
      this.pending.some((queued) => queued.id === id)
    );
  }

  // A request that reaches the server after its client gave up on it: the
  // route's claim, then the queue behind a running turn, or its own turn.
  private arrive(text: string, id: string) {
    if (this.took(id)) return;
    if (this.running()) this.pending.push({ text, id });
    else this.startTurn(text, this.queueing ? "queued" : "running", id);
  }

  // `is_message_duplicate`: the text of one of the trailing user rows.
  private repeatsTrailingPrompt(text: string) {
    for (let i = this.history.length - 1; i >= 0; i--) {
      if (this.history[i].role !== "user") return false;
      if (this.history[i].content === text) return true;
    }
    return false;
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

// `scoped_client_message_id`: the backend keys a send's row on its user,
// session and client `message_id`.
function scopedMessageId(url: URL, clientId: string) {
  const session = url.pathname.split("/").at(-2) ?? "";
  return persistedPromptId(USER_ID, session, clientId);
}

function hangUntilAborted(signal: AbortSignal | null | undefined) {
  return new Promise<Response>((_, reject) => {
    const abort = () =>
      reject(new DOMException("The operation was aborted.", "AbortError"));
    if (signal?.aborted) abort();
    else signal?.addEventListener("abort", abort, { once: true });
  });
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

// A body that fails once the client has read `delivered`.
function failsAfter(delivered: string) {
  let read = false;
  return new ReadableStream<Uint8Array>({
    pull(controller) {
      if (read) controller.error(new TypeError("network lost"));
      else controller.enqueue(encoder.encode(delivered));
      read = true;
    },
  });
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
