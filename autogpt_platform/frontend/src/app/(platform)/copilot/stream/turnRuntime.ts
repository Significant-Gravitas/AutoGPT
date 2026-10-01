import { getV2GetSession } from "@/app/api/__generated__/endpoints/chat/chat";
import { toast } from "@/components/molecules/Toast/use-toast";
import { environment } from "@/services/environment";
import * as Sentry from "@sentry/nextjs";
import type { FileUIPart, UIMessage } from "ai";
import { v4 as uuidv4 } from "uuid";

import {
  BACKLOG_DRAIN_TICKS,
  findWordCutPoints,
  TICK_DELAY_MS,
} from "../copilotStreamSmoothing";
import { streamRequestBody } from "../copilotStreamTransport";
import { getCopilotAuthHeaders, isEngineSwitchPart } from "../helpers";
import type { CopilotLlmModel } from "../store";
import { isTokenDevtoolEnabled } from "../tokenDevtool/gate";
import { createUsageCapturingFetch } from "../tokenDevtool/usageTap";
import { CANCELLED_MARKER } from "../useCopilotStop";
import { fetchSse, type SseFrame, type SseResult } from "./sseClient";
import { applyEntry, type StreamEntry, type WireChunk } from "./turnConverter";
import {
  COPILOT_RETRYABLE_ERROR_PREFIX,
  isMarker,
  logRowFromPersisted,
  seedTurnLog,
  type LogRow,
  type PersistedRow,
  type TurnLog,
} from "./turnLog";
import { diffRows, sha256Hex } from "./turnShadow";

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
const REVEAL_TICK_MS = 30;
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
const INTERRUPTED_MARKER = `${COPILOT_RETRYABLE_ERROR_PREFIX} Response was interrupted. Resend to try again.`;
const FROZEN_NOTE =
  "[__COPILOT_SYSTEM_e3b0__] This reply is shown as saved; it will update when the turn ends.";

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

export interface UserSegment {
  kind: "user";
  key: string;
  message: UIMessage;
  sequence: number | null;
  /** `sent` is the turn's own prompt; `chip` a promoted follow-up. */
  origin: "sent" | "chip";
  rawId: string | null;
}

export interface TurnSegment {
  kind: "turn";
  key: string;
  log: TurnLog;
  ended: boolean;
  stopped: boolean;
  /** Two drift events on one turn: the persisted view is shown instead. */
  frozen: boolean;
  reconciled: boolean;
  /** Fields that never stream, adopted at the reconcile. */
  durationMs: number | null;
  createdAt: string | null;
}

/** Persisted rows the tail holds but no turn of this mount produced. */
export interface RowsSegment {
  kind: "rows";
  key: string;
  rows: readonly PersistedRow[];
}

export type Segment = UserSegment | TurnSegment | RowsSegment;

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

interface TurnCheckpoint {
  entry_id: string;
  rows: number;
  sequence: number;
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
  private providerFailure: unknown = null;
  private lastViewMaxSequence: number | null = null;
  private turns = new Map<string, TurnBookkeeping>();
  private retryTimer: ReturnType<typeof setTimeout> | null = null;
  private tickTimer: ReturnType<typeof setInterval> | null = null;
  private frozenTimer: ReturnType<typeof setInterval> | null = null;
  private emitTimer: ReturnType<typeof setTimeout> | null = null;
  // Characters shown of a row the live POST stream is writing: smoothing
  // paces what is shown, never what is applied, so the cursor stays exact.
  private revealed = new Map<string, number>();
  private revealTimer: ReturnType<typeof setInterval> | null = null;
  private displayRows = new WeakMap<LogRow, { shown: number; row: LogRow }>();
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

  hasTurn(turnId: string) {
    return this.turnFor(turnId) !== null;
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
    this.segments.push({
      kind: "user",
      key: message.id,
      message,
      sequence: null,
      origin: "sent",
      rawId: null,
    });
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
    this.revealed.clear();
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
      this.segments.push({
        kind: "user",
        key,
        message: userMessage(key, { text: entry.text }),
        sequence: null,
        origin: "chip",
        rawId: null,
      });
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
    const rows = persistedRows(view);
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
      this.ensureConnected("view");
    }
    this.emitNow();
  }

  // ── Connection ─────────────────────────────────────────────────────────

  /**
   * Open a connection at the cursor unless one is open and delivered a frame
   * within the liveness window. One slot, so a second call inside the window
   * does nothing: two resumes never read one turn at once.
   */
  ensureConnected(_reason: string) {
    const running = this.runningTurn();
    if (!running?.log.turnId || running.frozen) return;
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
    for (const timer of [this.tickTimer, this.frozenTimer, this.revealTimer]) {
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
      const result: SseResult = await fetchSse(
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
    if (conn.frames === 1) this.resetFailures();
    if (frame.kind === "entry") this.applyFrom(conn, frame.entry);
    else if (frame.kind === "synthetic" && frame.chunk.type === "error") {
      conn.syntheticError = str(frame.chunk.errorText);
    }
    this.emit();
  }

  private applyFrom(conn: Connection, entry: StreamEntry) {
    let seg = this.turnFor(entry.turn);
    if (!seg) {
      if (conn.kind !== "post" || conn.turnId !== null) return;
      seg = this.openTurn(
        entry.turn,
        seedTurnLog({
          turnId: entry.turn,
          rows: [],
          checkpoint: null,
        }),
      );
    }
    conn.turnId = entry.turn;
    if (seg.ended || seg.stopped || seg.frozen) return;
    const next = applyEntry(seg.log, entry);
    if (next === seg.log) return;
    conn.entries += 1;
    const key = seg.key;
    this.trackReveal(conn, seg.log, next);
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
      void this.recoverUnknownTurn(conn, end);
      return;
    }
    if (end.kind === "refused") {
      if (end.status === 409) {
        void this.resync(seg.key, checkpointOf(end.body));
        return;
      }
      if (end.status === 410 || end.status === 204) {
        void this.expire(seg.key);
        return;
      }
      this.retry(conn);
      return;
    }
    // A clean close without the turn's finish is an expected cut (a proxy
    // restart, the route's own error frame): resume at once.
    if (end.kind === "closed" && conn.frames > 0) {
      this.resume(seg);
      return;
    }
    this.retry(conn);
  }

  /**
   * The disconnect classification (design §1.7). Hidden, offline and
   * rotation-age cuts resume silently; only a run of failures while the tab
   * is visible and online reaches the one alarming toast.
   */
  private retry(conn: Connection) {
    const running = this.runningTurn();
    if (!running && conn.turnId !== null) return;
    if (document.visibilityState === "hidden") return;
    if (typeof navigator !== "undefined" && navigator.onLine === false) {
      this.notice = "offline";
      this.emitNow();
      return;
    }
    if (Date.now() - conn.openedAt >= ROTATE_AFTER_MS && running) {
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
      if (this.runningTurn()) this.ensureConnected("retry");
      else void this.recoverUnknownTurn(conn, { kind: "closed" });
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
  // whether the turn runs.
  private async recoverUnknownTurn(conn: Connection, end: ConnectionEnd) {
    const view = await this.fetchView();
    if (!view) {
      this.retry(conn);
      return;
    }
    if (view.active_stream && this.wouldAttach(view)) {
      this.resetFailures();
      this.observe(view);
      return;
    }
    this.observe(view);
    if (end.kind === "failed" || conn.syntheticError) {
      this.error =
        end.kind === "failed" && end.error instanceof Error
          ? end.error
          : new Error(conn.syntheticError ?? "The connection closed early.");
      this.emitNow();
    }
  }

  // ── The turn's lifecycle ───────────────────────────────────────────────

  private attach(
    active: NonNullable<SessionView["active_stream"]>,
    view: SessionView,
  ) {
    const rows = persistedRows(view);
    const checkpoint = active.checkpoint ?? null;
    const start = checkpoint
      ? checkpoint.sequence
      : (this.lastViewMaxSequence ?? -1) + 1;
    this.fillGap(start, rows);
    this.claimTail(start);
    const seg = this.openTurn(
      active.turn_id,
      seedTurnLog({ turnId: active.turn_id, rows, checkpoint }),
    );
    this.resume(seg);
  }

  private endTurn(key: string) {
    this.updateTurn(key, (seg) => ({ ...seg, ended: true }));
    this.resetFailures();
    this.emitNow();
    this.handlers.onTurnEnd?.();
    const seg = this.segmentByKey(key);
    if (!seg || seg.kind !== "turn") return;
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
    const view = await this.catchUp(() => this.fetchView());
    const seg = this.segmentByKey(key);
    if (!seg || seg.kind !== "turn" || !seg.log.turnId) return;
    if (!view) {
      book.resyncs -= 1;
      this.retry(this.deadConnection(seg.log.turnId));
      return;
    }
    const active = view.active_stream;
    const at =
      checkpoint ??
      (active?.turn_id === seg.log.turnId ? active.checkpoint : null) ??
      lastCheckpoint(seg.log);
    const seeded = seedTurnLog({
      turnId: seg.log.turnId,
      rows: persistedRows(view),
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
  private async expire(key: string) {
    const view = await this.catchUp(() => this.fetchView());
    const seg = this.segmentByKey(key);
    if (!seg || seg.kind !== "turn") return;
    if (!view) {
      this.retry(this.deadConnection(seg.log.turnId));
      return;
    }
    const persisted = this.turnRows(seg, persistedRows(view));
    const open =
      Object.values(seg.log.blocks).some((b) => b.open) ||
      Object.values(seg.log.tools).some((t) => t.phase !== "output-available");
    this.updateTurn(key, (s) => {
      const rows = persisted ? mergeRows(s.log.rows, persisted) : s.log.rows;
      const last = rows[rows.length - 1];
      const interrupted = open && !(last && isMarker(last));
      return {
        ...s,
        ended: true,
        reconciled: persisted !== null,
        log: {
          ...closeBlocks(s.log),
          status: "finished",
          rows: interrupted
            ? [
                ...rows,
                markerRow(`interrupted:${s.log.turnId}`, INTERRUPTED_MARKER),
              ]
            : rows,
        },
      };
    });
    this.resetFailures();
    this.emitNow();
    this.handlers.onTurnEnd?.();
    this.observe(view);
  }

  private async freeze(key: string) {
    this.abortSlot();
    this.clearRetry();
    this.deps.report("frozen", { turnId: this.turnIdOf(key) });
    const view = await this.fetchView();
    this.updateTurn(key, (seg) => {
      const persisted = view ? this.turnRows(seg, persistedRows(view)) : null;
      return {
        ...seg,
        frozen: true,
        log: {
          ...closeBlocks(seg.log),
          rows: [
            ...(persisted ?? seg.log.rows),
            markerRow(`frozen:${seg.log.turnId}`, FROZEN_NOTE),
          ],
        },
      };
    });
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

  /**
   * The end-of-turn reconcile: the turn's persisted rows against its log, row
   * by row. Equal rows adopt their sequence and keep their object; a differing
   * row is replaced under its own key and reported; a streamed row the DB
   * lacks stays on screen.
   */
  private reconcile(seg: TurnSegment, view: SessionView) {
    const rows = persistedRows(view);
    const start = this.turnStart(seg, rows);
    if (start === null) return;
    const windowStart = rows[0]?.sequence ?? Infinity;
    if (view.has_more_messages !== false && windowStart > start) return;
    const persisted = this.turnRows(seg, rows);
    if (!persisted) return;
    const drift = diffRows(seg.log.rows, persisted);
    if (drift.length > 0 && !seg.stopped && !seg.frozen) {
      this.deps.report("finish", {
        turnId: seg.log.turnId,
        cursor: seg.log.cursor,
        diffs: drift,
      });
    }
    const raw = rows.filter(
      (r) =>
        (r.sequence ?? -1) >= start &&
        (r.sequence ?? -1) < start + persisted.length,
    );
    const durations = raw
      .map((r) => (r as { duration_ms?: unknown }).duration_ms)
      .filter((d): d is number => typeof d === "number");
    this.updateTurn(seg.key, (s) => ({
      ...s,
      reconciled: true,
      ended: true,
      log: {
        ...closeBlocks(s.log),
        rows: s.frozen ? persisted : mergeRows(s.log.rows, persisted),
      },
      durationMs: durations.length ? Math.max(...durations) : s.durationMs,
      createdAt: createdAtOf(raw[raw.length - 1]) ?? s.createdAt,
    }));
    this.adoptPromptSequence(seg.key, start, rows);
  }

  // ── Tail bookkeeping ───────────────────────────────────────────────────

  private turnStart(seg: TurnSegment, rows: readonly PersistedRow[]) {
    const fromCheckpoint = seg.log.checkpoints[0]?.sequence;
    if (typeof fromCheckpoint === "number") return fromCheckpoint;
    const seeded = seg.log.rows[0]?.sequence;
    if (typeof seeded === "number") return seeded;
    const index = this.segments.indexOf(seg);
    const before = this.segments[index - 1];
    if (before?.kind === "user") {
      const prompt =
        before.sequence ?? locatePrompt(before, rows, this.ownedFrom);
      return prompt === null ? null : prompt + 1;
    }
    return index === 0 ? this.ownedFrom : null;
  }

  // A turn's rows run from its start to the next segment that has a place.
  private turnRows(seg: TurnSegment, rows: readonly PersistedRow[]) {
    const start = this.turnStart(seg, rows);
    if (start === null) return null;
    const index = this.segments.indexOf(seg);
    const next = this.segments
      .slice(index + 1)
      .map((s) => segmentStart(s))
      .find((n): n is number => n !== null);
    return rows
      .filter(
        (r) =>
          typeof r.sequence === "number" &&
          r.sequence >= start &&
          r.sequence < (next ?? Infinity),
      )
      .map(logRowFromPersisted);
  }

  private adoptPromptSequence(
    key: string,
    start: number,
    rows: readonly PersistedRow[],
  ) {
    const index = this.segments.findIndex((s) => s.key === key);
    const before = this.segments[index - 1];
    if (before?.kind !== "user" || before.sequence !== null) return;
    const row = rows.find((r) => r.sequence === start - 1);
    if (row?.role !== "user") return;
    const rawId = (row as { id?: unknown }).id;
    this.segments[index - 1] = {
      ...before,
      sequence: start - 1,
      rawId: typeof rawId === "string" ? rawId : null,
    };
  }

  // Rows another tab (or the server) persisted between this tail's end and a
  // turn it now attaches to.
  private fillGap(start: number, rows: readonly PersistedRow[]) {
    if (this.ownedFrom === null) return;
    const tailEnd = this.tailEndSequence();
    if (tailEnd === null) return;
    const gap = rows.filter(
      (r) => (r.sequence ?? -1) > tailEnd && (r.sequence ?? -1) < start,
    );
    if (gap.length === 0) return;
    this.segments.push({
      kind: "rows",
      key: `rows:${gap[0].sequence}`,
      rows: gap,
    });
  }

  private tailEndSequence() {
    for (let i = this.segments.length - 1; i >= 0; i--) {
      const seg = this.segments[i];
      if (seg.kind === "rows")
        return seg.rows[seg.rows.length - 1]?.sequence ?? null;
      if (seg.kind === "user") return seg.sequence;
      const last = seg.log.rows[seg.log.rows.length - 1];
      return last?.sequence ?? null;
    }
    return null;
  }

  private claimTail(from: number) {
    if (this.ownedFrom === null || from < this.ownedFrom) this.ownedFrom = from;
  }

  private nextSequence() {
    return this.ownedFrom !== null && this.segments.length > 0
      ? (this.tailEndSequence() ?? this.ownedFrom) + 1
      : (this.lastViewMaxSequence ?? -1) + 1;
  }

  private openTurn(turnId: string, log: TurnLog): TurnSegment {
    const seg: TurnSegment = {
      kind: "turn",
      key: `turn:${turnId}`,
      log,
      ended: false,
      stopped: false,
      frozen: false,
      reconciled: false,
      durationMs: null,
      createdAt: null,
    };
    this.segments.push(seg);
    return seg;
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

  // ── Paced text ─────────────────────────────────────────────────────────

  private trackReveal(conn: Connection, prev: TurnLog, next: TurnLog) {
    if (conn.kind !== "post") {
      this.revealed.clear();
      return;
    }
    for (const block of Object.values(next.blocks)) {
      if (!block.open || block.row === null) continue;
      const row = next.rows[block.row];
      if (!this.revealed.has(row.key)) {
        this.revealed.set(row.key, prev.rows[block.row]?.content.length ?? 0);
      }
    }
    if (this.revealed.size > 0 && !this.revealTimer) {
      this.revealTimer = setInterval(() => this.revealTick(), REVEAL_TICK_MS);
    }
  }

  // Each tick shows what the smoothing transform would have emitted over it:
  // a share of the backlog per 10 ms, a whole word at least.
  private revealTick() {
    const open = this.openRows();
    for (const [key, shown] of this.revealed) {
      const row = open.get(key);
      if (!row) {
        this.revealed.delete(key);
        continue;
      }
      let next = shown;
      for (let t = 0; t < REVEAL_TICK_MS / TICK_DELAY_MS; t++) {
        const cuts = findWordCutPoints(row.content.slice(next));
        if (cuts.length === 0) break;
        const words = Math.max(1, Math.ceil(cuts.length / BACKLOG_DRAIN_TICKS));
        next += cuts[Math.min(words, cuts.length) - 1];
      }
      if (next !== shown) this.revealed.set(key, next);
    }
    if (this.revealed.size === 0 && this.revealTimer) {
      clearInterval(this.revealTimer);
      this.revealTimer = null;
    }
    this.emitNow();
  }

  private openRows() {
    const rows = new Map<string, LogRow>();
    for (const seg of this.segments) {
      if (seg.kind !== "turn") continue;
      for (const block of Object.values(seg.log.blocks)) {
        if (!block.open || block.row === null) continue;
        const row = seg.log.rows[block.row];
        if (row) rows.set(row.key, row);
      }
    }
    return rows;
  }

  private display(seg: Segment): Segment {
    if (seg.kind !== "turn" || this.revealed.size === 0) return seg;
    let changed = false;
    const rows = seg.log.rows.map((row) => {
      const shown = this.revealed.get(row.key);
      if (shown === undefined || shown >= row.content.length) return row;
      changed = true;
      const cached = this.displayRows.get(row);
      if (cached?.shown === shown) return cached.row;
      const display = { ...row, content: row.content.slice(0, shown) };
      this.displayRows.set(row, { shown, row: display });
      return display;
    });
    return changed ? { ...seg, log: { ...seg.log, rows } } : seg;
  }

  // ── Plumbing ───────────────────────────────────────────────────────────

  private onVisible = () => {
    if (document.visibilityState !== "visible") return;
    this.ensureConnected("wake");
  };

  private onOnline = () => {
    if (this.notice === "offline") this.notice = null;
    this.resetFailures();
    this.ensureConnected("online");
    this.emitNow();
  };

  private onOffline = () => {
    if (this.runningTurn() && !this.isLive()) {
      this.notice = "offline";
      this.emitNow();
    }
  };

  private ensureTicking() {
    if (this.tickTimer) return;
    this.tickTimer = setInterval(() => this.tick(), TICK_MS);
  }

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
      if (slot.turnId === null)
        void this.recoverUnknownTurn(slot, { kind: "closed" });
      else this.retry(slot);
      return;
    }
    if (!this.rotating && now - slot.openedAt >= ROTATE_AFTER_MS) this.rotate();
  }

  private async catchUp<T>(work: () => Promise<T>) {
    const timer = setTimeout(() => {
      this.notice = "catching-up";
      this.emitNow();
    }, CATCHING_UP_AFTER_MS);
    try {
      return await work();
    } finally {
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

  private turnIdOf(key: string) {
    const seg = this.segmentByKey(key);
    return seg?.kind === "turn" ? seg.log.turnId : null;
  }

  private updateTurn(key: string, patch: (seg: TurnSegment) => TurnSegment) {
    const index = this.segments.findIndex((s) => s.key === key);
    const seg = this.segments[index];
    if (seg?.kind !== "turn") return;
    this.segments[index] = patch(seg);
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

  private deadConnection(turnId: string | null): Connection {
    const now = Date.now();
    return {
      kind: "resume",
      controller: new AbortController(),
      openedAt: now,
      lastFrameAt: now,
      frames: 0,
      entries: 0,
      turnId,
      closedByClient: false,
      syntheticError: null,
    };
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
      segments: this.segments.map((seg) => this.display(seg)),
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
    const last = [...this.segments]
      .reverse()
      .find((s): s is TurnSegment => s.kind === "turn");
    if (this.error) return "failed";
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

/** Rows equal to the ones on screen keep their object and key; others take the persisted content under the on-screen key. */
export function mergeRows(
  live: readonly LogRow[],
  persisted: readonly LogRow[],
): LogRow[] {
  const differing = new Set(diffRows(live, persisted).map((d) => d.index));
  const merged: LogRow[] = [];
  for (let i = 0; i < Math.max(live.length, persisted.length); i++) {
    const a = live[i];
    const b = persisted[i];
    if (!b) merged.push(a);
    else if (!a) merged.push(b);
    else if (differing.has(i)) merged.push({ ...b, key: a.key });
    else merged.push(adoptPersisted(a, b));
  }
  return merged;
}

// A resync's seed keeps the rows already on screen where they are equal.
function keepIdentity(live: readonly LogRow[], seeded: readonly LogRow[]) {
  const differing = new Set(
    diffRows(live.slice(0, seeded.length), seeded).map((d) => d.index),
  );
  return seeded.map((row, i) => {
    const current = live[i];
    if (!current) return row;
    return differing.has(i)
      ? { ...row, key: current.key }
      : adoptPersisted(current, row);
  });
}

function adoptPersisted(live: LogRow, persisted: LogRow): LogRow {
  const sameMetadata =
    JSON.stringify(live.metadata ?? null) ===
    JSON.stringify(persisted.metadata ?? live.metadata ?? null);
  if (live.sequence === persisted.sequence && sameMetadata) return live;
  return {
    ...live,
    sequence: persisted.sequence,
    metadata: persisted.metadata ?? live.metadata,
  };
}

function withStopMarker(log: TurnLog): TurnLog {
  const last = log.rows[log.rows.length - 1];
  const rows =
    last && isMarker(last)
      ? log.rows
      : [...log.rows, markerRow(`stop:${log.turnId}`, CANCELLED_MARKER)];
  return { ...closeBlocks(log), status: "finished", rows };
}

function closeBlocks(log: TurnLog): TurnLog {
  const open = Object.entries(log.blocks).filter(([, b]) => b.open);
  if (open.length === 0) return log;
  const blocks = { ...log.blocks };
  for (const [id, block] of open) blocks[id] = { ...block, open: false };
  return { ...log, blocks };
}

function markerRow(key: string, content: string): LogRow {
  return {
    key,
    role: "assistant",
    content,
    toolCalls: [],
    toolCallId: null,
    sequence: null,
    metadata: null,
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

function persistedRows(view: SessionView): PersistedRow[] {
  return (view.messages ?? [])
    .map((row) => row as PersistedRow)
    .filter((row) => typeof row.sequence === "number")
    .sort((a, b) => (a.sequence ?? 0) - (b.sequence ?? 0));
}

function segmentStart(seg: Segment): number | null {
  if (seg.kind === "user") return seg.sequence;
  if (seg.kind === "rows") return seg.rows[0]?.sequence ?? null;
  return seg.log.checkpoints[0]?.sequence ?? seg.log.rows[0]?.sequence ?? null;
}

// The prompt's row: the first user row in the tail whose text it carries
// (the backend appends an attached-files block after it).
function locatePrompt(
  seg: UserSegment,
  rows: readonly PersistedRow[],
  from: number | null,
) {
  const text = seg.message.parts
    .map((p) => (p.type === "text" ? p.text : ""))
    .join("");
  const row = rows.find(
    (r) =>
      r.role === "user" &&
      (r.sequence ?? -1) >= (from ?? 0) &&
      typeof r.content === "string" &&
      r.content.startsWith(text),
  );
  return row?.sequence ?? null;
}

function lastCheckpoint(log: TurnLog): TurnCheckpoint | null {
  const last = log.checkpoints[log.checkpoints.length - 1];
  return last
    ? { entry_id: last.entryId, rows: last.rows, sequence: last.sequence }
    : null;
}

function checkpointOf(body: unknown): TurnCheckpoint | null {
  const checkpoint = (body as { checkpoint?: unknown } | null)?.checkpoint;
  if (!checkpoint || typeof checkpoint !== "object") return null;
  const { entry_id, rows, sequence } = checkpoint as Record<string, unknown>;
  return typeof entry_id === "string" &&
    typeof rows === "number" &&
    typeof sequence === "number"
    ? { entry_id, rows, sequence }
    : null;
}

function createdAtOf(row: PersistedRow | undefined): string | null {
  const value = (row as { created_at?: unknown } | undefined)?.created_at;
  if (typeof value === "string") return value;
  return value instanceof Date ? value.toISOString() : null;
}

function str(value: unknown): string {
  return typeof value === "string" ? value : "";
}

function sleep(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
