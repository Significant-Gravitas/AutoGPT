import { getV2GetSession } from "@/app/api/__generated__/endpoints/chat/chat";
import { toast } from "@/components/molecules/Toast/use-toast";
import { environment } from "@/services/environment";
import * as Sentry from "@sentry/nextjs";
import type { FileUIPart, UIMessage } from "ai";
import { v4 as uuidv4 } from "uuid";

import { streamRequestBody } from "../copilotStreamTransport";
import { getCopilotAuthHeaders, isEngineSwitchPart } from "../helpers";
import type { CopilotLlmModel } from "../store";
import { isTokenDevtoolEnabled } from "../tokenDevtool/gate";
import { createUsageCapturingFetch } from "../tokenDevtool/usageTap";
import { fetchSse, type SseFrame } from "./sseClient";
import { TextReveal } from "./textReveal";
import { applyEntry, type StreamEntry, type WireChunk } from "./turnConverter";
import { seedTurnLog } from "./turnLog";
import { diffRows, sha256Hex } from "./turnShadow";
import {
  checkpointOf,
  closeBlocks,
  FROZEN_NOTE,
  gapSegment,
  hasOpenParts,
  INTERRUPTED_MARKER,
  keepIdentity,
  lastCheckpoint,
  locatePrompt,
  mergeRows,
  newTurnSegment,
  persistedRows,
  reconcileTurn,
  tailEndSequence,
  turnRows,
  userSegment,
  withMarker,
  withPromptRow,
  withStopMarker,
  type Segment,
  type TurnCheckpoint,
  type TurnSegment,
} from "./turnTail";

/** No byte for this long means the connection is dead: the route and the
 *  listener heartbeat every 10 s and the SDK engine every 10 s of silence. */
export const LIVENESS_MS = 30_000;
/** The load balancer cuts an SSE at 30 minutes; rotate before it does. */
export const ROTATE_AFTER_MS = 25 * 60_000;
const BACKOFF_MS = [1_000, 2_000, 4_000, 8_000, 16_000];
const ANOMALY_RETRY_MS = 30_000;
const INDICATOR_AFTER_FAILURES = 2;
const CATCHING_UP_AFTER_MS = 2_000;
const TICK_MS = 5_000;
const EMIT_THROTTLE_MS = 30;
const FROZEN_POLL_MS = 10_000;
const FINISH_PROBE_MS = 500;
// A server-started continuation (engine switch, approval wake) is dispatched
// after the turn ends; its meta can lag the finish by a few seconds.
const CONTINUATION_PROBES = 8;
const SETTLE_PROBES = 3;
const MAX_RESYNCS_PER_TURN = 1;
// Resyncing on an orphan tool output, which the backend does persist, would
// loop; only a fold that lost track of a block or a call is rebuilt.
const RESYNC_REASONS = new Set([
  "unknown block",
  "closed block",
  "input changed",
]);

export const ANOMALY_TOAST = {
  title: "Connection lost",
  description: "We keep trying; your expert may still be working.",
};

export type RuntimePhase =
  | "idle"
  | "connecting"
  | "live"
  | "resuming"
  | "finished"
  | "failed";

/** A passive, inline state; never a toast. */
export type RuntimeNotice = "catching-up" | "reconnecting" | "offline" | null;

export interface RuntimeSnapshot {
  /** The tail this runtime owns, in order; history before it is the DB view's. */
  segments: readonly Segment[];
  /** First DB sequence the tail owns; null while it owns nothing. */
  ownedFrom: number | null;
  phase: RuntimePhase;
  notice: RuntimeNotice;
  error: Error | null;
  stopped: boolean;
}

export interface SessionView {
  messages?: readonly unknown[] | null;
  has_more_messages?: boolean;
  active_stream?: {
    turn_id: string;
    checkpoint?: TurnCheckpoint | null;
  } | null;
}

export interface RuntimeDeps {
  baseUrl: () => string;
  headers: () => Promise<Record<string, string>>;
  fetch: typeof fetch;
  fetchSession: () => Promise<SessionView | null>;
  toast: (options: { title: string; description: string }) => void;
  report: (kind: string, extra: Record<string, unknown>) => void;
}

export interface RuntimeHandlers {
  /** A turn's error entry or a refused send, with the provider failure sent before it. */
  onError?: (error: Error, providerFailure: unknown) => void;
  onTurnEnd?: () => void;
}

export type SendInput =
  | { text: string; files?: FileUIPart[]; metadata?: unknown }
  | { parts: UIMessage["parts"]; metadata?: unknown };

interface Connection {
  kind: "post" | "resume";
  controller: AbortController;
  openedAt: number;
  lastFrameAt: number;
  frames: number;
  entries: number;
  turnId: string | null;
  closedByClient: boolean;
  syntheticError: string | null;
}

/** What the classification needs to know about a connection that is gone. */
type Lost = Pick<Connection, "turnId" | "openedAt" | "syntheticError">;

type ConnectionEnd =
  | { kind: "closed" }
  | { kind: "refused"; status: number; body: unknown }
  | { kind: "failed"; error: unknown };

interface TurnBookkeeping {
  verified: number;
  reportedErrors: number;
  resyncs: number;
  sawModeChange: boolean;
}

/**
 * The chat's connection to a session's running turn: one connection slot,
 * one cursor per turn, and a tail of segments rendered from the rows the
 * backend persists. Survives unmounts in a module-level map, as the chat
 * runtime it replaces did, so switching sessions does not drop a stream.
 */
export class TurnRuntime {
  private segments: Segment[] = [];
  private ownedFrom: number | null = null;
  private slot: Connection | null = null;
  private rotating: Connection | null = null;
  private failures = 0;
  private anomalyShown = false;
  private notice: RuntimeNotice = null;
  private error: Error | null = null;
  private stopped = false;
  // A stopped turn keeps running on the server for a moment; never re-attach to it.
  private suppressAttach = false;
  private followPending = false;
  // A resync or an expiry is rebuilding the turn; it reconnects when done.
  private recovering = false;
  // A send whose stream ended before naming its turn, while hidden or offline.
  private unresolvedPost: { lost: Lost; end: ConnectionEnd } | null = null;
  private providerFailure: unknown = null;
  private lastViewMaxSequence: number | null = null;
  private turns = new Map<string, TurnBookkeeping>();
  private retryTimer: ReturnType<typeof setTimeout> | null = null;
  private tickTimer: ReturnType<typeof setInterval> | null = null;
  private frozenTimer: ReturnType<typeof setInterval> | null = null;
  private emitTimer: ReturnType<typeof setTimeout> | null = null;
  private reveal = new TextReveal(
    () => this.segments,
    () => this.emitNow(),
  );
  private listeners = new Set<() => void>();
  private snapshot: RuntimeSnapshot;
  private handlers: RuntimeHandlers = {};
  private boundFetchSession: (() => Promise<SessionView | null>) | null = null;
  private disposed = false;

  constructor(
    readonly sessionId: string,
    private readonly deps: RuntimeDeps,
  ) {
    this.snapshot = this.buildSnapshot();
    if (typeof window !== "undefined") {
      document.addEventListener("visibilitychange", this.onVisible);
      window.addEventListener("pageshow", this.onVisible);
      document.addEventListener("resume", this.onVisible);
      window.addEventListener("online", this.onOnline);
      window.addEventListener("offline", this.onOffline);
    }
  }

  subscribe = (listener: () => void) => {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  };

  getSnapshot = () => this.snapshot;

  /** The mounted chat's callbacks and session fetch; an unmounted runtime falls back. */
  bind(
    handlers: RuntimeHandlers,
    fetchSession: (() => Promise<SessionView | null>) | null,
  ) {
    this.handlers = handlers;
    this.boundFetchSession = fetchSession;
    return () => {
      if (this.handlers === handlers) this.handlers = {};
      if (this.boundFetchSession === fetchSession)
        this.boundFetchSession = null;
    };
  }

  /** Whether `observe(view)` would attach to the view's running turn. */
  wouldAttach(view: SessionView) {
    const active = view.active_stream;
    return (
      !!active &&
      !this.turnFor(active.turn_id) &&
      !this.suppressAttach &&
      !this.isPostPending() &&
      !this.runningTurn()
    );
  }

  // ── The five named operations ─────────────────────────────────────────

  /** POST a new turn: the prompt shows at once, and the turn follows it. */
  async send(input: SendInput, model: CopilotLlmModel | undefined) {
    if (this.isPostPending()) return;
    this.stopped = false;
    this.suppressAttach = false;
    this.error = null;
    this.providerFailure = null;
    const message = userMessage(`local:${uuidv4({})}`, input);
    this.claimTail(this.nextSequence());
    this.segments.push(userSegment(message, "sent"));
    this.emitNow();
    const body = streamRequestBody(this.sessionId, message, model);
    await this.open("post", null, this.streamUrl(), {
      method: "POST",
      body: JSON.stringify(body),
      extraHeaders: { "Content-Type": "application/json" },
    });
  }

  /** Stop following the turn: abort, mark it stopped, and let the persisted marker reconcile into it. */
  stop() {
    const running = this.runningTurn();
    this.abortConnections();
    this.clearRetry();
    this.stopped = true;
    this.suppressAttach = true;
    this.notice = null;
    this.reveal.clear();
    if (running) {
      this.updateTurn(running.key, (seg) => ({
        ...seg,
        stopped: true,
        log: withStopMarker(seg.log),
      }));
    }
    this.emitNow();
  }

  /** Promoted follow-ups; a drained row in the turn with the same text stands in for one. */
  appendLocalUserRows(entries: readonly { id: string; text: string }[]) {
    if (entries.length === 0) return;
    this.claimTail(this.nextSequence());
    for (const entry of entries) {
      const key = `pending-chip-${entry.id}`;
      if (this.segments.some((s) => s.key === key)) continue;
      this.segments.push(
        userSegment(userMessage(key, { text: entry.text }), "chip"),
      );
    }
    this.emit();
  }

  /** The backend refused the send before persisting it (a 429). */
  dropUnsentUserRow() {
    const last = this.segments[this.segments.length - 1];
    if (last?.kind !== "user" || last.origin !== "sent") return;
    this.segments.pop();
    if (this.segments.length === 0) this.ownedFrom = null;
    this.emitNow();
  }

  /**
   * Every fresh session view: reconcile the turns it has finished, attach
   * to a turn it runs that this runtime does not follow, and fetch the rest
   * of a running turn the server no longer runs.
   */
  observe(view: SessionView) {
    const rows = persistedRows(view.messages);
    const last = rows[rows.length - 1]?.sequence;
    if (typeof last === "number") this.lastViewMaxSequence = last;
    const active = view.active_stream ?? null;
    if (!active) {
      this.suppressAttach = false;
      this.stopped = false;
    }
    for (const seg of this.segments) {
      if (seg.kind !== "turn" || seg.reconciled) continue;
      if (active?.turn_id === seg.log.turnId) continue;
      if (seg.ended || seg.stopped || seg.frozen) this.reconcile(seg, view);
    }
    if (active && this.wouldAttach(view)) this.attach(active, view);
    const running = this.runningTurn();
    if (running && active?.turn_id !== running.log.turnId && !running.frozen) {
      this.ensureConnected();
    }
    if (!running && !this.isPostPending() && !active) this.settleTail(view);
    this.emitNow();
  }

  // ── Connection ─────────────────────────────────────────────────────────

  /**
   * Open a connection at the cursor unless one is open and delivered a frame
   * within the liveness window. One slot, so a second call inside the window
   * does nothing: two resumes never read one turn at once.
   */
  ensureConnected() {
    const running = this.runningTurn();
    if (!running?.log.turnId || running.frozen || this.recovering) return;
    if (this.slot && Date.now() - this.slot.lastFrameAt < LIVENESS_MS) return;
    this.clearRetry();
    this.resume(running);
  }

  /** An answered approval starts the chat's next turn on the server; attach to it. */
  followBackendTurn() {
    if (this.runningTurn() || this.isPostPending()) {
      this.followPending = true;
      return;
    }
    void this.probe(null, CONTINUATION_PROBES, true);
  }

  dispose() {
    this.disposed = true;
    this.abortConnections();
    this.clearRetry();
    this.reveal.dispose();
    for (const timer of [this.tickTimer, this.frozenTimer]) {
      if (timer) clearInterval(timer);
    }
    if (this.emitTimer) clearTimeout(this.emitTimer);
    if (typeof window !== "undefined") {
      document.removeEventListener("visibilitychange", this.onVisible);
      window.removeEventListener("pageshow", this.onVisible);
      document.removeEventListener("resume", this.onVisible);
      window.removeEventListener("online", this.onOnline);
      window.removeEventListener("offline", this.onOffline);
    }
  }

  isIdle() {
    return !this.slot && !this.runningTurn() && !this.isPostPending();
  }

  private resume(seg: TurnSegment) {
    const turnId = seg.log.turnId;
    if (!turnId) return;
    this.abortSlot();
    void this.open("resume", turnId, this.resumeUrl(turnId, seg.log.cursor));
  }

  // Rotation opens the second connection first and retires the first on the
  // second's first frame; entries both deliver are dropped by the cursor.
  private rotate() {
    const running = this.runningTurn();
    const turnId = running?.log.turnId;
    if (!running || !turnId || this.rotating) return;
    void this.open(
      "resume",
      turnId,
      this.resumeUrl(turnId, running.log.cursor),
      {},
      true,
    );
  }

  private async open(
    kind: Connection["kind"],
    turnId: string | null,
    url: string,
    init: { method?: string; body?: string; extraHeaders?: object } = {},
    asRotation = false,
  ) {
    const now = Date.now();
    const conn: Connection = {
      kind,
      controller: new AbortController(),
      openedAt: now,
      lastFrameAt: now,
      frames: 0,
      entries: 0,
      turnId,
      closedByClient: false,
      syntheticError: null,
    };
    if (asRotation) this.rotating = conn;
    else {
      this.abortSlot();
      this.slot = conn;
    }
    this.ensureTicking();
    this.emit();
    let end: ConnectionEnd;
    try {
      const headers = await this.deps.headers();
      const result = await fetchSse(
        url,
        {
          method: init.method ?? "GET",
          body: init.body,
          headers: { ...headers, ...init.extraHeaders },
          signal: conn.controller.signal,
        },
        (frame) => this.onFrame(conn, frame),
        this.deps.fetch,
      );
      end = result.ok
        ? { kind: "closed" }
        : { kind: "refused", status: result.status, body: result.body };
    } catch (error) {
      end = { kind: "failed", error };
    }
    this.onEnd(conn, end);
  }

  private onFrame(conn: Connection, frame: SseFrame) {
    if (conn !== this.slot && conn !== this.rotating) return;
    conn.lastFrameAt = Date.now();
    conn.frames += 1;
    if (conn === this.rotating && conn.frames === 1) {
      const retired = this.slot;
      this.slot = conn;
      this.rotating = null;
      retired?.controller.abort();
      if (retired) retired.closedByClient = true;
    }
    // Every response ends with [DONE]; only an entry or a heartbeat shows the
    // route is following a running turn.
    if (frame.kind === "entry" || frame.kind === "comment")
      this.resetFailures();
    if (frame.kind === "entry") {
      conn.entries += 1;
      this.applyFrom(conn, frame.entry);
    } else if (frame.kind === "synthetic" && frame.chunk.type === "error") {
      conn.syntheticError = str(frame.chunk.errorText);
    }
    this.emit();
  }

  private applyFrom(conn: Connection, entry: StreamEntry) {
    let seg = this.turnFor(entry.turn);
    if (!seg) {
      if (conn.kind !== "post" || conn.turnId !== null) return;
      seg = newTurnSegment(
        entry.turn,
        seedTurnLog({ turnId: entry.turn, rows: [], checkpoint: null }),
      );
      this.segments.push(seg);
    }
    conn.turnId = entry.turn;
    if (seg.ended || seg.stopped || seg.frozen) return;
    const next = applyEntry(seg.log, entry);
    if (next === seg.log) return;
    if (conn.kind === "post") this.reveal.track(seg.log, next);
    else this.reveal.clear();
    const key = seg.key;
    this.updateTurn(key, (s) => ({ ...s, log: next }));
    this.onChunk(key, entry.chunk);
    this.checkProtocol(key);
    void this.verifyCheckpoints(key);
    // The turn is over at an id-bearing finish, and failed at an id-bearing error.
    if (entry.chunk.type === "finish" || entry.chunk.type === "error") {
      this.endTurn(key);
    }
  }

  private onChunk(key: string, chunk: WireChunk) {
    if (chunk.type === "data-provider-failure")
      this.providerFailure = chunk.data;
    if (isEngineSwitchPart(chunk)) this.bookkeeping(key).sawModeChange = true;
    if (chunk.type === "error") {
      const failure = this.providerFailure;
      this.providerFailure = null;
      this.handlers.onError?.(new Error(str(chunk.errorText)), failure);
    }
  }

  private onEnd(conn: Connection, end: ConnectionEnd) {
    if (conn === this.rotating) {
      this.rotating = null;
      return;
    }
    if (conn !== this.slot || this.disposed) return;
    this.slot = null;
    this.emit();
    if (conn.closedByClient) return;
    const seg = conn.turnId ? this.turnFor(conn.turnId) : null;
    if (seg && (seg.ended || seg.stopped || seg.frozen)) return;
    if (end.kind === "refused" && conn.kind === "post") {
      this.refusedSend(end);
      return;
    }
    if (!seg) {
      this.findLostTurn(conn, end);
      return;
    }
    if (end.kind === "refused") {
      if (end.status === 409) void this.resync(seg.key, checkpointOf(end.body));
      else if (end.status === 410 || end.status === 204)
        void this.expire(seg.key);
      else this.retry(conn);
      return;
    }
    // A clean close after entries but without the turn's finish is an expected
    // cut (a proxy restart, the route's own error frame): resume at once.
    if (end.kind === "closed" && conn.entries > 0) this.resume(seg);
    else if (end.kind === "closed") void this.closedEmpty(seg.key, conn);
    else this.retry(conn);
  }

  // A read that yields no entry may be of a turn the server stopped running
  // without storing its finish; resuming it again would loop.
  private async closedEmpty(key: string, lost: Lost) {
    const view = await this.recover(() => this.fetchView());
    const seg = this.segmentByKey(key);
    if (seg?.kind !== "turn" || seg.ended || seg.stopped || seg.frozen) return;
    if (view && view.active_stream?.turn_id !== seg.log.turnId)
      void this.expire(key, view);
    else this.retry(lost);
  }

  /**
   * The disconnect classification (design §1.7). Hidden, offline and
   * rotation-age cuts resume silently; only a run of failures while the tab
   * is visible and online reaches the one alarming toast.
   */
  private retry(lost: Lost) {
    const running = this.runningTurn();
    if (!running && lost.turnId !== null) return;
    const offline =
      typeof navigator !== "undefined" && navigator.onLine === false;
    if (
      lost.turnId === null &&
      (offline || document.visibilityState === "hidden")
    ) {
      this.unresolvedPost = { lost, end: { kind: "closed" } };
    }
    if (document.visibilityState === "hidden") return;
    if (offline) {
      this.notice = "offline";
      this.emitNow();
      return;
    }
    if (Date.now() - lost.openedAt >= ROTATE_AFTER_MS && running) {
      this.resume(running);
      return;
    }
    this.failures += 1;
    if (this.failures >= INDICATOR_AFTER_FAILURES) this.notice = "reconnecting";
    if (this.failures >= BACKOFF_MS.length && !this.anomalyShown) {
      this.anomalyShown = true;
      this.deps.toast(ANOMALY_TOAST);
    }
    const delay = BACKOFF_MS[this.failures - 1] ?? ANOMALY_RETRY_MS;
    this.clearRetry();
    this.retryTimer = setTimeout(() => {
      this.retryTimer = null;
      if (this.runningTurn()) this.ensureConnected();
      else void this.recoverUnknownTurn(lost, { kind: "closed" });
    }, delay);
    this.emitNow();
  }

  private refusedSend(end: Extract<ConnectionEnd, { kind: "refused" }>) {
    const text =
      typeof end.body === "string" ? end.body : JSON.stringify(end.body);
    this.error = new Error(text || `Request failed with status ${end.status}`);
    this.emitNow();
    this.handlers.onError?.(this.error, null);
  }

  // A send whose stream ended before naming its turn: the session view says
  // whether the turn runs, once the tab can ask.
  private findLostTurn(lost: Lost, end: ConnectionEnd) {
    const offline =
      typeof navigator !== "undefined" && navigator.onLine === false;
    if (document.visibilityState === "hidden" || offline)
      this.unresolvedPost = { lost, end };
    else void this.recoverUnknownTurn(lost, end);
  }

  // The turn registers a beat after the POST answers, so the view is asked a
  // few times before the send is reported as failed.
  private async recoverUnknownTurn(lost: Lost, end: ConnectionEnd) {
    for (let i = 0; i < CONTINUATION_PROBES; i++) {
      const view = await this.fetchView();
      if (!view) {
        this.retry(lost);
        return;
      }
      const attaching = this.wouldAttach(view);
      this.observe(view);
      if (attaching) {
        this.resetFailures();
        return;
      }
      if (this.disposed || this.stopped) return;
      await sleep(FINISH_PROBE_MS);
    }
    if (end.kind === "failed" || lost.syntheticError) {
      this.error =
        end.kind === "failed" && end.error instanceof Error
          ? end.error
          : new Error(lost.syntheticError ?? "The connection closed early.");
      this.emitNow();
    }
  }

  // ── The turn's lifecycle ───────────────────────────────────────────────

  private attach(
    active: NonNullable<SessionView["active_stream"]>,
    view: SessionView,
  ) {
    const rows = persistedRows(view.messages);
    const checkpoint = active.checkpoint ?? null;
    const start = checkpoint
      ? checkpoint.sequence
      : (this.lastViewMaxSequence ?? -1) + 1;
    // Rows another tab (or the server) persisted between the tail and this turn.
    const gap =
      this.ownedFrom === null ? null : gapSegment(this.segments, start, rows);
    if (gap) this.segments.push(gap);
    this.claimTail(start);
    const seg = newTurnSegment(
      active.turn_id,
      seedTurnLog({ turnId: active.turn_id, rows, checkpoint }),
      start,
    );
    this.segments.push(seg);
    this.resume(seg);
  }

  private endTurn(key: string) {
    this.updateTurn(key, (seg) => ({ ...seg, ended: true }));
    this.resetFailures();
    this.emitNow();
    this.handlers.onTurnEnd?.();
    const seg = this.segmentByKey(key);
    if (seg?.kind !== "turn") return;
    const continuation =
      this.bookkeeping(key).sawModeChange || this.followPending;
    this.followPending = false;
    void this.probe(
      seg.log.turnId,
      continuation ? CONTINUATION_PROBES : SETTLE_PROBES,
      continuation,
    );
  }

  /**
   * Fetch the session view until it no longer runs `turnId`: the end-of-turn
   * reconcile, and the attach to a continuation it starts.
   */
  private async probe(
    turnId: string | null,
    attempts: number,
    awaitContinuation: boolean,
  ) {
    for (let i = 0; i < attempts; i++) {
      await sleep(FINISH_PROBE_MS);
      if (this.disposed || this.stopped) return;
      const view = await this.fetchView();
      if (!view) continue;
      this.observe(view);
      const active = view.active_stream?.turn_id ?? null;
      if (active && active !== turnId) return;
      if (!active && !awaitContinuation) return;
    }
  }

  /**
   * Rebuild the turn from the DB rows at a checkpoint and tail the stream
   * from there: a trimmed stream, a protocol error, a digest mismatch. A
   * second one on the same turn freezes it instead, so a real bug cannot
   * become a reconnect storm.
   */
  private async resync(key: string, checkpoint: TurnCheckpoint | null) {
    const book = this.bookkeeping(key);
    if (book.resyncs >= MAX_RESYNCS_PER_TURN) {
      void this.freeze(key);
      return;
    }
    book.resyncs += 1;
    this.abortSlot();
    const view = await this.recover(() => this.fetchView());
    const seg = this.segmentByKey(key);
    if (seg?.kind !== "turn" || !seg.log.turnId) return;
    if (!view) {
      book.resyncs -= 1;
      this.retry({
        turnId: seg.log.turnId,
        openedAt: Date.now(),
        syntheticError: null,
      });
      return;
    }
    const active = view.active_stream;
    const at =
      checkpoint ??
      (active?.turn_id === seg.log.turnId ? active.checkpoint : null) ??
      lastCheckpoint(seg.log);
    const seeded = seedTurnLog({
      turnId: seg.log.turnId,
      rows: persistedRows(view.messages),
      checkpoint: at,
    });
    book.verified = 0;
    book.reportedErrors = 0;
    this.updateTurn(key, (s) => ({
      ...s,
      log: { ...seeded, rows: keepIdentity(s.log.rows, seeded.rows) },
    }));
    const next = this.segmentByKey(key);
    if (next?.kind === "turn") this.resume(next);
  }

  // The stream is gone: the DB view is the whole truth, and a turn that
  // stopped mid-part renders as interrupted.
  private async expire(key: string, known: SessionView | null = null) {
    const view = known ?? (await this.recover(() => this.fetchView()));
    const seg = this.segmentByKey(key);
    if (seg?.kind !== "turn") return;
    if (!view) {
      this.retry({
        turnId: seg.log.turnId,
        openedAt: Date.now(),
        syntheticError: null,
      });
      return;
    }
    const persisted = turnRows(
      this.segments,
      this.ownedFrom,
      seg,
      persistedRows(view.messages),
    );
    const interrupted = hasOpenParts(seg.log);
    this.updateTurn(key, (s) => {
      const rows = persisted
        ? mergeRows(s.log.rows, persisted.rows)
        : s.log.rows;
      return {
        ...s,
        ended: true,
        reconciled: persisted !== null,
        log: {
          ...closeBlocks(s.log),
          status: "finished",
          rows: interrupted
            ? withMarker(
                rows,
                `interrupted:${s.log.turnId}`,
                INTERRUPTED_MARKER,
              )
            : rows,
        },
      };
    });
    this.resetFailures();
    this.emitNow();
    this.handlers.onTurnEnd?.();
    this.observe(view);
  }

  // Show the persisted rows with a note, and poll until the turn is over.
  private async freeze(key: string) {
    this.abortSlot();
    this.clearRetry();
    const view = await this.fetchView();
    this.updateTurn(key, (seg) => {
      const persisted = view
        ? turnRows(
            this.segments,
            this.ownedFrom,
            seg,
            persistedRows(view.messages),
          )
        : null;
      return {
        ...seg,
        frozen: true,
        log: {
          ...closeBlocks(seg.log),
          rows: withMarker(
            persisted?.rows ?? seg.log.rows,
            `frozen:${seg.log.turnId}`,
            FROZEN_NOTE,
          ),
        },
      };
    });
    this.deps.report("frozen", { turnId: key });
    this.notice = null;
    this.emitNow();
    if (this.frozenTimer) clearInterval(this.frozenTimer);
    this.frozenTimer = setInterval(async () => {
      const next = await this.fetchView();
      if (next) this.observe(next);
      const seg = this.segmentByKey(key);
      if (seg?.kind === "turn" && seg.reconciled && this.frozenTimer) {
        clearInterval(this.frozenTimer);
        this.frozenTimer = null;
      }
    }, FROZEN_POLL_MS);
  }

  private reconcile(seg: TurnSegment, view: SessionView) {
    const rows = persistedRows(view.messages);
    const persisted = turnRows(this.segments, this.ownedFrom, seg, rows);
    if (!persisted) return;
    // The session GET returns a window; a turn starting before it waits.
    const windowStart = rows[0]?.sequence ?? Infinity;
    if (view.has_more_messages !== false && windowStart > persisted.start)
      return;
    const drift = diffRows(seg.log.rows, persisted.rows);
    if (drift.length > 0 && !seg.stopped && !seg.frozen) {
      this.deps.report("finish", {
        turnId: seg.log.turnId,
        cursor: seg.log.cursor,
        diffs: drift,
      });
    }
    this.updateTurn(seg.key, (s) => reconcileTurn(s, persisted));
    const index = this.segments.findIndex((s) => s.key === seg.key);
    const prompt = this.segments[index - 1];
    const row = rows.find((r) => r.sequence === persisted.start - 1);
    if (
      prompt?.kind === "user" &&
      prompt.sequence === null &&
      row?.role === "user"
    ) {
      this.segments[index - 1] = withPromptRow(prompt, row);
    }
  }

  // With nothing running, a prompt whose turn never reached this client (a
  // stop before its first entry, a lost send) takes its row, and rows after
  // the tail's end are shown as persisted.
  private settleTail(view: SessionView) {
    const rows = persistedRows(view.messages);
    const last = this.segments[this.segments.length - 1];
    if (
      last?.kind === "user" &&
      last.origin === "sent" &&
      last.sequence === null
    ) {
      const row = locatePrompt(last, rows, this.ownedFrom);
      if (!row) return;
      this.segments[this.segments.length - 1] = withPromptRow(last, row);
    }
    const gap =
      this.ownedFrom === null
        ? null
        : gapSegment(this.segments, Infinity, rows);
    if (gap) this.segments.push(gap);
  }

  private claimTail(from: number) {
    if (this.ownedFrom === null || from < this.ownedFrom) this.ownedFrom = from;
  }

  private nextSequence() {
    return this.ownedFrom !== null && this.segments.length > 0
      ? (tailEndSequence(this.segments) ?? this.ownedFrom) + 1
      : (this.lastViewMaxSequence ?? -1) + 1;
  }

  private checkProtocol(key: string) {
    const seg = this.segmentByKey(key);
    if (seg?.kind !== "turn") return;
    const book = this.bookkeeping(key);
    const fresh = seg.log.protocolErrors.slice(book.reportedErrors);
    book.reportedErrors = seg.log.protocolErrors.length;
    for (const error of fresh) {
      this.deps.report("protocol_error", {
        turnId: seg.log.turnId,
        cursor: seg.log.cursor,
        ...error,
      });
    }
    if (fresh.some((e) => RESYNC_REASONS.has(e.reason))) {
      void this.resync(key, null);
    }
  }

  private async verifyCheckpoints(key: string) {
    const book = this.bookkeeping(key);
    let seg = this.segmentByKey(key);
    while (seg?.kind === "turn" && book.verified < seg.log.checkpoints.length) {
      const checkpoint = seg.log.checkpoints[book.verified++];
      const digest = await sha256Hex(checkpoint.canonical);
      seg = this.segmentByKey(key);
      if (digest === null || digest === checkpoint.digest) continue;
      if (seg?.kind !== "turn") return;
      this.deps.report("checkpoint", {
        turnId: seg.log.turnId,
        entryId: checkpoint.entryId,
        rows: checkpoint.rows,
        serverDigest: checkpoint.digest,
        clientDigest: digest,
      });
      void this.resync(key, {
        entry_id: checkpoint.entryId,
        rows: checkpoint.rows,
        sequence: checkpoint.sequence,
      });
      return;
    }
  }

  // ── Plumbing ───────────────────────────────────────────────────────────

  private onVisible = () => {
    if (document.visibilityState === "visible") this.wake();
  };

  private onOnline = () => {
    if (this.notice === "offline") this.notice = null;
    this.resetFailures();
    this.wake();
    this.emitNow();
  };

  private onOffline = () => {
    if (this.runningTurn() && !this.isLive()) {
      this.notice = "offline";
      this.emitNow();
    }
  };

  private wake() {
    const unresolved = this.unresolvedPost;
    this.unresolvedPost = null;
    if (unresolved && !this.runningTurn())
      void this.recoverUnknownTurn(unresolved.lost, unresolved.end);
    else this.ensureConnected();
  }

  private ensureTicking() {
    if (this.tickTimer) return;
    this.tickTimer = setInterval(() => this.tick(), TICK_MS);
  }

  // Liveness and rotation; a hidden tab's timers are not to be trusted.
  private tick() {
    const slot = this.slot;
    if (!slot && !this.runningTurn()) {
      if (this.tickTimer) clearInterval(this.tickTimer);
      this.tickTimer = null;
      return;
    }
    if (!slot || document.visibilityState === "hidden") return;
    const now = Date.now();
    if (now - slot.lastFrameAt >= LIVENESS_MS) {
      slot.closedByClient = true;
      slot.controller.abort();
      this.slot = null;
      if (slot.turnId === null) this.findLostTurn(slot, { kind: "closed" });
      else this.retry(slot);
      return;
    }
    if (!this.rotating && now - slot.openedAt >= ROTATE_AFTER_MS) this.rotate();
  }

  // Rebuilding from the DB view: no other trigger reconnects meanwhile, and
  // a rebuild past 2 s shows an inline "catching up".
  private async recover<T>(work: () => Promise<T>) {
    this.recovering = true;
    const timer = setTimeout(() => {
      this.notice = "catching-up";
      this.emitNow();
    }, CATCHING_UP_AFTER_MS);
    try {
      return await work();
    } finally {
      this.recovering = false;
      clearTimeout(timer);
      if (this.notice === "catching-up") this.notice = null;
      this.emit();
    }
  }

  private async fetchView() {
    try {
      return await (this.boundFetchSession ?? this.deps.fetchSession)();
    } catch {
      return null;
    }
  }

  private runningTurn(): TurnSegment | null {
    for (let i = this.segments.length - 1; i >= 0; i--) {
      const seg = this.segments[i];
      if (seg.kind !== "turn") continue;
      return seg.ended || seg.stopped ? null : seg;
    }
    return null;
  }

  private isPostPending() {
    return this.slot?.kind === "post" && this.slot.turnId === null;
  }

  private isLive() {
    return (
      !!this.slot &&
      this.slot.frames > 0 &&
      Date.now() - this.slot.lastFrameAt < LIVENESS_MS
    );
  }

  private turnFor(turnId: string): TurnSegment | null {
    const seg = this.segments.find(
      (s): s is TurnSegment => s.kind === "turn" && s.log.turnId === turnId,
    );
    return seg ?? null;
  }

  private segmentByKey(key: string) {
    return this.segments.find((s) => s.key === key) ?? null;
  }

  private updateTurn(key: string, patch: (seg: TurnSegment) => TurnSegment) {
    const index = this.segments.findIndex((s) => s.key === key);
    const seg = this.segments[index];
    if (seg?.kind === "turn") this.segments[index] = patch(seg);
  }

  private bookkeeping(key: string) {
    let book = this.turns.get(key);
    if (!book) {
      book = {
        verified: 0,
        reportedErrors: 0,
        resyncs: 0,
        sawModeChange: false,
      };
      this.turns.set(key, book);
    }
    return book;
  }

  private resetFailures() {
    this.failures = 0;
    this.anomalyShown = false;
    if (this.notice === "reconnecting" || this.notice === "offline") {
      this.notice = null;
    }
  }

  private abortSlot() {
    const slot = this.slot;
    this.slot = null;
    if (!slot) return;
    slot.closedByClient = true;
    slot.controller.abort();
  }

  private abortConnections() {
    this.abortSlot();
    const rotating = this.rotating;
    this.rotating = null;
    if (rotating) {
      rotating.closedByClient = true;
      rotating.controller.abort();
    }
  }

  private clearRetry() {
    if (this.retryTimer) clearTimeout(this.retryTimer);
    this.retryTimer = null;
  }

  private streamUrl() {
    return `${this.deps.baseUrl()}/api/chat/sessions/${this.sessionId}/stream`;
  }

  private resumeUrl(turnId: string, cursor: string | null) {
    const params = new URLSearchParams({
      turn: turnId,
      after: cursor ?? "0-0",
    });
    return `${this.streamUrl()}?${params}`;
  }

  private emit() {
    if (this.emitTimer) return;
    this.emitTimer = setTimeout(() => {
      this.emitTimer = null;
      this.emitNow();
    }, EMIT_THROTTLE_MS);
  }

  private emitNow() {
    if (this.emitTimer) clearTimeout(this.emitTimer);
    this.emitTimer = null;
    this.snapshot = this.buildSnapshot();
    for (const listener of this.listeners) listener();
  }

  private buildSnapshot(): RuntimeSnapshot {
    return {
      segments: this.segments.map((seg) => this.reveal.display(seg)),
      ownedFrom: this.ownedFrom,
      phase: this.phase(),
      notice: this.notice,
      error: this.error,
      stopped: this.stopped,
    };
  }

  private phase(): RuntimePhase {
    if (this.isPostPending()) return "connecting";
    const running = this.runningTurn();
    if (running) return this.isLive() ? "live" : "resuming";
    if (this.error) return "failed";
    const last = this.segments.findLast(
      (s): s is TurnSegment => s.kind === "turn",
    );
    if (!last || last.stopped) return "idle";
    return last.log.status === "failed" ? "failed" : "finished";
  }
}

const runtimes = new Map<string, TurnRuntime>();
// A runtime holds its turns' rows; idle ones beyond this are dropped.
const MAX_RUNTIMES = 5;

export function getTurnRuntime(
  sessionId: string,
  deps: Partial<RuntimeDeps> = {},
) {
  const existing = runtimes.get(sessionId);
  if (existing) {
    runtimes.delete(sessionId);
    runtimes.set(sessionId, existing);
    return existing;
  }
  const runtime = new TurnRuntime(sessionId, {
    ...defaultDeps(sessionId),
    ...deps,
  });
  runtimes.set(sessionId, runtime);
  for (const [id, other] of runtimes) {
    if (runtimes.size <= MAX_RUNTIMES) break;
    if (id !== sessionId && other.isIdle()) {
      other.dispose();
      runtimes.delete(id);
    }
  }
  return runtime;
}

export function resetTurnRuntimes() {
  for (const runtime of runtimes.values()) runtime.dispose();
  runtimes.clear();
}

function defaultDeps(sessionId: string): RuntimeDeps {
  return {
    baseUrl: () => environment.getAGPTServerBaseUrl(),
    headers: getCopilotAuthHeaders,
    // The token devtool reads the `: usage` comments off the same response.
    fetch: isTokenDevtoolEnabled()
      ? createUsageCapturingFetch(sessionId)
      : (input, init) => fetch(input, init),
    fetchSession: async () => {
      const response = await getV2GetSession(sessionId);
      return response.status === 200 ? (response.data as SessionView) : null;
    },
    toast: (options) => toast(options),
    report: (kind, extra) =>
      Sentry.captureMessage(`copilot stream drift: ${kind}`, {
        level: "warning",
        tags: { stream_drift: kind },
        extra: { sessionId, ...extra },
      }),
  };
}

function userMessage(id: string, input: SendInput): UIMessage {
  const parts =
    "parts" in input
      ? input.parts
      : [
          ...(input.files ?? []),
          ...(input.text
            ? [
                {
                  type: "text" as const,
                  text: input.text,
                  state: "done" as const,
                },
              ]
            : []),
        ];
  return {
    id,
    role: "user",
    parts,
    ...(input.metadata ? { metadata: input.metadata } : {}),
  };
}

function str(value: unknown): string {
  return typeof value === "string" ? value : "";
}

function sleep(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
