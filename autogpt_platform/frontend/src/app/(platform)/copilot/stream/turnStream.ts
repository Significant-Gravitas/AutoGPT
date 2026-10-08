import type { UIMessageChunk } from "ai";

import { readSseFrames, type SseFrame } from "./sseClient";
import {
  applyEntry,
  compareEntryIds,
  type StreamEntry,
  type WireChunk,
} from "./turnConverter";
import { isRenderable } from "./renderable";
import { emptyTurnLog, rowsDigest, type TurnLog } from "./turnLog";
import { observeShadowEnd, observeShadowEntry } from "./turnShadow";

/**
 * One turn as the chat renders it, across however many connections it takes.
 *
 * The AI SDK parser reading `readable` lives for the whole turn: a dropped
 * connection is reopened here with `?turn=&after=<cursor>`, every entry at or
 * before the cursor is dropped, and the parser only ever sees each entry once
 * and never a part whose start it missed. What to do about a lost connection
 * (when to reconnect, when to give up) is decided by the stream lifecycle;
 * this only reports it and carries the command out.
 */
export type TurnStreamPhase =
  "connecting" | "open" | "lost" | "finished" | "closed";

export interface TurnStreamState {
  phase: TurnStreamPhase;
  turnId: string | null;
  /** Once finished: the folded rows reproduce the turn's last checkpoint, so
   *  the screen already shows what the database holds. */
  verified: boolean | null;
  /** Connections lost in a row; back to 0 once a frame arrives. */
  failures: number;
}

interface TurnStreamArgs {
  sessionId: string;
  /** Known up front for a resume; a send learns it from its first entry. */
  turnId: string | null;
  /** True for a send: its refusal is the AI SDK's error to surface. */
  isSend: boolean;
  /** Keep the chat's last message id when the parser continues into it. */
  dropStartMessageId: boolean;
  openFirst: (signal: AbortSignal) => Promise<Response>;
  /** GET the stream with `query` (empty for the route's legacy replay). */
  openResume: (query: string, signal: AbortSignal) => Promise<Response>;
  onComment?: (text: string) => void;
}

export class TurnStream {
  readonly readable: ReadableStream<UIMessageChunk>;
  private log: TurnLog = emptyTurnLog();
  private state: TurnStreamState;
  private finishing = false;
  /** Entries were skipped (a trimmed stream): the fold cannot vouch for the rows. */
  private gap = false;
  private legacyResume = false;
  /** Id-less chunks forwarded so far, and how many the current connection replayed. */
  private legacyForwarded = 0;
  private legacyReplayed = 0;
  private frameAt = Date.now();
  private connection: AbortController | null = null;
  private downstream: ReadableStreamDefaultController<UIMessageChunk> | null =
    null;
  private listeners = new Set<() => void>();

  constructor(private readonly args: TurnStreamArgs) {
    this.state = {
      phase: "connecting",
      turnId: args.turnId,
      verified: null,
      failures: 0,
    };
    this.readable = new ReadableStream<UIMessageChunk>({
      start: (controller) => {
        this.downstream = controller;
      },
      cancel: () => this.close(),
    });
  }

  getState(): TurnStreamState {
    return this.state;
  }

  /** When the last frame (heartbeats included) or connection attempt happened. */
  get lastFrameAt() {
    return this.frameAt;
  }

  subscribe(listener: () => void) {
    this.listeners.add(listener);
    return () => {
      this.listeners.delete(listener);
    };
  }

  /**
   * Open the first connection. Resolves with the chunk stream once it
   * answers, or null when a resume finds nothing to read. A send's refusal
   * rejects, as the AI SDK expects of a transport.
   */
  async open(): Promise<ReadableStream<UIMessageChunk> | null> {
    const outcome = await this.connect(this.args.openFirst, true);
    return outcome === "empty" ? null : this.readable;
  }

  /** Drop the current connection, if any, and read on from the cursor. */
  reconnect() {
    if (this.isOver()) return;
    void this.connect((signal) => this.openResume(signal), false).catch(() =>
      this.connectionLost(),
    );
  }

  /** Give up on the turn's stream; the chat hydrates it from the database. */
  abandon() {
    if (this.isOver()) return;
    this.end(false);
  }

  /** Close the connection from this side. The turn runs on on the server. */
  close() {
    if (this.isOver()) return;
    observeShadowEnd(this.args.sessionId, this.state.turnId, true);
    this.dropConnection();
    this.closeDownstream();
    this.set({ phase: "closed" });
  }

  private isOver() {
    return (
      this.finishing ||
      this.state.phase === "finished" ||
      this.state.phase === "closed"
    );
  }

  private async connect(
    open: (signal: AbortSignal) => Promise<Response>,
    first: boolean,
  ): Promise<"open" | "empty"> {
    const abort = new AbortController();
    this.dropConnection();
    this.connection = abort;
    this.frameAt = Date.now();
    this.set({ phase: "connecting" });
    let response: Response;
    try {
      response = await open(abort.signal);
    } catch (error) {
      if (first && this.args.isSend) throw error;
      if (abort === this.connection && !this.isOver()) this.connectionLost();
      return "open";
    }
    if (abort !== this.connection || this.isOver()) {
      void response.body?.cancel().catch(() => {});
      return "open";
    }
    if (!response.ok || !response.body) {
      if (first && this.args.isSend) {
        throw new Error(
          (await response.text().catch(() => "")) ||
            "Failed to fetch the chat response.",
        );
      }
      return this.refused(response, first);
    }
    this.frameAt = Date.now();
    this.set({ phase: "open" });
    this.legacyReplayed = 0;
    void this.read(response.body, abort);
    return "open";
  }

  private async refused(
    response: Response,
    first: boolean,
  ): Promise<"open" | "empty"> {
    if (response.status === 409) {
      const body = (await response.json().catch(() => null)) as {
        checkpoint?: { entry_id?: unknown } | null;
      } | null;
      const entryId = body?.checkpoint?.entry_id;
      const cursor = this.log.cursor;
      this.gap = true;
      if (
        typeof entryId === "string" &&
        (cursor === null || compareEntryIds(entryId, cursor) > 0)
      ) {
        // The stream from the checkpoint on is whole: read it, and leave the
        // rows the trim took to the end-of-turn hydrate.
        this.log = { ...this.log, cursor: entryId };
      } else {
        this.legacyResume = true;
      }
      return this.connect((signal) => this.openResume(signal), first);
    }
    if ([204, 404, 410].includes(response.status)) {
      this.end(false);
      return first ? "empty" : "open";
    }
    this.connectionLost();
    return "open";
  }

  private openResume(signal: AbortSignal) {
    const turnId = this.state.turnId;
    const query =
      this.legacyResume || turnId === null
        ? ""
        : `?turn=${encodeURIComponent(turnId)}&after=${this.log.cursor ?? "0-0"}`;
    return this.args.openResume(query, signal);
  }

  private async read(body: ReadableStream<Uint8Array>, abort: AbortController) {
    try {
      await readSseFrames(
        body,
        (frame) => {
          if (abort === this.connection && !this.isOver()) this.onFrame(frame);
        },
        abort.signal,
      );
    } catch {
      // A broken body is a lost connection like any other.
    }
    if (abort !== this.connection || this.isOver()) return;
    this.connectionLost();
  }

  private onFrame(frame: SseFrame) {
    this.frameAt = Date.now();
    if (this.state.failures > 0) this.set({ failures: 0 });
    switch (frame.kind) {
      case "comment":
        this.args.onComment?.(frame.text);
        return;
      case "entry":
        this.applyEntryFrame(frame.entry);
        return;
      case "synthetic":
        this.applySynthetic(frame.chunk);
        return;
      case "done":
        return;
    }
  }

  private applyEntryFrame(entry: StreamEntry) {
    const turnId = this.state.turnId;
    if (turnId !== null && entry.turn !== turnId) return;
    const cursor = this.log.cursor;
    if (cursor !== null && compareEntryIds(entry.entryId, cursor) <= 0) return;
    const before = this.log;
    this.log = applyEntry(before, entry);
    if (turnId === null) this.set({ turnId: entry.turn });
    observeShadowEntry(this.args.sessionId, entry);
    if (isRenderable(before, entry.chunk))
      this.emit(this.forParser(entry.chunk));
    if (entry.chunk.type === "finish") void this.finish();
  }

  // A stream that carries no entry ids (an older route) is forwarded whole; a
  // replay of it is skipped up to what was already forwarded. Once ids flow,
  // an id-less data frame is one the route made up: it ends this connection,
  // never the turn.
  private applySynthetic(chunk: WireChunk) {
    if (this.log.cursor === null) {
      if (this.legacyReplayed++ < this.legacyForwarded) return;
      this.legacyForwarded += 1;
      this.emit(this.forParser(chunk));
      if (chunk.type === "finish") void this.finish();
      return;
    }
    if (chunk.type === "finish" || chunk.type === "error") {
      this.connection?.abort();
    }
  }

  private async finish() {
    if (this.isOver()) return;
    this.finishing = true;
    const verified = await this.verify().catch(() => false);
    this.finishing = false;
    this.end(verified);
  }

  private async verify() {
    if (this.gap) return false;
    const last = this.log.checkpoints[this.log.checkpoints.length - 1];
    if (!last || last.rows !== this.log.rows.length) return false;
    return (await rowsDigest(this.log.rows)) === last.digest;
  }

  private end(verified: boolean) {
    this.dropConnection();
    this.closeDownstream();
    this.set({ phase: "finished", verified });
  }

  private connectionLost() {
    observeShadowEnd(this.args.sessionId, this.state.turnId, false);
    this.connection = null;
    this.set({ phase: "lost", failures: this.state.failures + 1 });
  }

  private dropConnection() {
    const connection = this.connection;
    this.connection = null;
    connection?.abort();
  }

  private forParser(chunk: WireChunk): WireChunk {
    if (chunk.type !== "start" || !this.args.dropStartMessageId) return chunk;
    const { messageId: _dropped, ...rest } = chunk;
    return { ...rest, type: "start" };
  }

  private emit(chunk: WireChunk) {
    try {
      this.downstream?.enqueue(chunk as unknown as UIMessageChunk);
    } catch {
      // The reader went away; the turn is over for this chat.
    }
  }

  private closeDownstream() {
    const downstream = this.downstream;
    this.downstream = null;
    try {
      downstream?.close();
    } catch {
      // Already closed or cancelled.
    }
  }

  private set(patch: Partial<TurnStreamState>) {
    this.state = { ...this.state, ...patch };
    this.listeners.forEach((listener) => listener());
  }
}
