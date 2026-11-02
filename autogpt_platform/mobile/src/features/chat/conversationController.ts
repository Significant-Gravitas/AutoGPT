import {
  AppState,
  type AppStateStatus,
  type NativeEventSubscription,
} from "react-native";

import type { BackendClient } from "../../api/backend";
import {
  cancelTurn,
  createSession,
  describeChatError,
  getSessionDetail,
  openTurnStream,
  parseCheckpoint,
  resumeTurnStream,
  type SessionDetail,
} from "../../api/chat";
import type { HttpError } from "../../api/errors";
import type { OpenResponse } from "../../api/http";
import { newMessageID } from "../../lib/ids";
import { delay } from "../../lib/time";
import { splitActiveTurn } from "./model/activeTurn";
import { applyEntry } from "./model/applyEntry";
import { splitAttachments } from "./model/attachments";
import { mergeRows, rowsFromPersisted, type Row } from "./model/rows";
import {
  historyItems,
  liveItems,
  localItem,
  type LocalMessage,
  type TimelineItem,
} from "./model/timeline";
import { emptyTurnLog, seedTurnLog, type TurnLog } from "./model/turnLog";
import { runConnection, type ConnectionResult } from "./turnConnection";

export type TurnPhase =
  | "idle"
  | "sending"
  | "streaming"
  | "reconnecting"
  | "stopping"
  | "queued"
  | "offline";

export interface ConversationSnapshot {
  sessionID: string | null;
  expertID: string | null;
  loadState: "loading" | "ready" | "error";
  loadError: string | null;
  items: TimelineItem[];
  hasOlder: boolean;
  loadingOlder: boolean;
  refreshing: boolean;
  turn: TurnPhase;
  statusText: string | null;
  notice: string | null;
}

const PAGE_SIZE = 50;
const MAX_PAGE_SIZE = 200;
const BACKOFF_MS = [1_000, 2_000, 4_000, 8_000, 16_000];
const MAX_RECONNECTS = 8;
const EMPTY_RESUME_PAUSE_MS = 2_000;
const QUEUED_POLL_MS = 5_000;
const QUEUED_POLLS = 60;
const STOP_SETTLE_MS = 4_000;
const EMIT_THROTTLE_MS = 50;
const FOREGROUND_RECONNECT_AFTER_MS = 5_000;

type Opener = (signal: AbortSignal) => Promise<OpenResponse>;

/**
 * Owns one conversation for the life of its screen: persisted history, at
 * most one running turn, and messages sent from this device.
 *
 * Sends are never replayed automatically. A send that fails, or whose
 * delivery can't be confirmed, stays visible as failed until the user
 * retries it with the same idempotency key or discards it.
 */
export class ConversationController {
  private sessionID: string | null;
  private expertID: string | null;
  private readonly onSessionCreated: (sessionID: string) => void;
  private history: Row[] = [];
  private historyCache: {
    history: Row[];
    running: boolean;
    items: TimelineItem[];
  } | null = null;
  private belongsToTurn: ((row: Row) => boolean) | null = null;
  private oldestSequence: number | null = null;
  private hasOlder = false;
  private live: TurnLog | null = null;
  private replayedUserContents: string[] = [];
  private locals: LocalMessage[] = [];
  private loadState: ConversationSnapshot["loadState"];
  private loadError: string | null = null;
  private loadingOlder = false;
  private refreshing = false;
  private turn: TurnPhase = "idle";
  private notice: string | null = null;
  private epoch = 0;
  private connection: AbortController | null = null;
  private wake = new AbortController();
  private backgroundedAt: number | null = null;
  private appState: NativeEventSubscription | null = null;
  private disposed = false;
  private listeners = new Set<() => void>();
  private emitTimer: ReturnType<typeof setTimeout> | null = null;
  private snapshot: ConversationSnapshot;

  constructor(
    private readonly api: BackendClient,
    args: {
      sessionID: string | null;
      expertID: string | null;
      onSessionCreated: (id: string) => void;
    },
  ) {
    this.sessionID = args.sessionID;
    this.expertID = args.expertID;
    this.onSessionCreated = args.onSessionCreated;
    this.loadState = args.sessionID ? "loading" : "ready";
    this.snapshot = this.buildSnapshot();
  }

  subscribe = (listener: () => void) => {
    this.listeners.add(listener);
    return () => {
      this.listeners.delete(listener);
    };
  };

  getSnapshot = () => this.snapshot;

  start() {
    this.disposed = false;
    this.appState?.remove();
    this.appState = AppState.addEventListener("change", (state) =>
      this.onAppStateChange(state),
    );
    if (this.loadState === "ready") void this.refresh();
    else void this.load();
  }

  /** Stops listening. The server keeps running any turn; reopening resumes it. */
  dispose() {
    this.disposed = true;
    this.epoch += 1;
    this.connection?.abort();
    this.connection = null;
    this.rouse();
    this.appState?.remove();
    this.appState = null;
    if (this.emitTimer) clearTimeout(this.emitTimer);
    this.emitTimer = null;
    if (this.turn !== "idle") this.turn = "offline";
  }

  canSend(): boolean {
    return this.loadState === "ready" && this.turn === "idle";
  }

  /** Retries a conversation that failed to open. */
  reload() {
    if (this.loadState === "error") void this.load();
  }

  async send(text: string) {
    const message = text.trim();
    if (!message || !this.canSend()) return;
    const local: LocalMessage = {
      messageID: newMessageID(),
      text: message,
      state: "sending",
      error: null,
      createdAt: Date.now(),
    };
    this.locals = [...this.locals, local];
    await this.deliver(local);
  }

  /** Re-sends a failed message with its original idempotency key. */
  async retry(messageID: string) {
    const local = this.locals.find(
      (l) => l.messageID === messageID && l.state === "failed",
    );
    if (!local || !this.canSend()) return;
    this.patchLocal(messageID, { state: "sending", error: null });
    await this.deliver(local);
  }

  discard(messageID: string) {
    this.locals = this.locals.filter(
      (l) => !(l.messageID === messageID && l.state === "failed"),
    );
    this.emitNow();
  }

  async stop() {
    const sessionID = this.sessionID;
    if (
      !sessionID ||
      !["streaming", "reconnecting", "queued", "offline"].includes(this.turn)
    )
      return;
    const previous = this.turn;
    this.turn = "stopping";
    this.emitNow();
    try {
      await cancelTurn(this.api, sessionID);
    } catch (error) {
      if (this.disposed) return;
      this.turn = previous;
      this.notice = `Couldn't stop: ${describeChatError(error)}`;
      this.emitNow();
      return;
    }
    // Give the stream a moment to deliver the turn's finish, then settle from history.
    const epoch = this.epoch;
    await delay(STOP_SETTLE_MS, this.wake.signal);
    if (this.disposed || !this.isStopping() || epoch !== this.epoch) return;
    await this.finishTurn(this.beginEpoch());
  }

  /** Pull to refresh, foregrounding, or a manual reconnect after going offline. */
  async refresh() {
    const sessionID = this.sessionID;
    if (
      !sessionID ||
      this.refreshing ||
      this.turn === "sending" ||
      this.turn === "stopping"
    )
      return;
    if (this.turn === "streaming" || this.turn === "reconnecting") {
      // Resume from the cursor straight away instead of waiting out a dead socket.
      this.connection?.abort();
      this.rouse();
      return;
    }
    this.refreshing = true;
    this.notice = null;
    this.emitNow();
    const epoch = this.beginEpoch();
    try {
      const detail = await getSessionDetail(this.api, sessionID, {
        limit: PAGE_SIZE,
      });
      if (this.isCurrent(epoch))
        await this.adoptServerState(detail, epoch, null);
    } catch (error) {
      if (this.isCurrent(epoch)) this.notice = describeChatError(error);
    } finally {
      this.refreshing = false;
      if (!this.disposed) this.emitNow();
    }
  }

  async loadOlder() {
    const sessionID = this.sessionID;
    if (
      !sessionID ||
      !this.hasOlder ||
      this.loadingOlder ||
      this.oldestSequence === null
    )
      return;
    this.loadingOlder = true;
    this.emitNow();
    try {
      const detail = await getSessionDetail(this.api, sessionID, {
        limit: PAGE_SIZE,
        beforeSequence: this.oldestSequence,
      });
      if (this.disposed) return;
      this.history = mergeRows(
        this.withoutTurnRows(rowsFromPersisted(detail.messages)),
        this.history,
      );
      this.oldestSequence = detail.oldestSequence ?? this.oldestSequence;
      this.hasOlder = detail.hasMore;
    } catch (error) {
      if (!this.disposed)
        this.notice = `Couldn't load earlier messages. ${describeChatError(error)}`;
    } finally {
      this.loadingOlder = false;
      if (!this.disposed) this.emitNow();
    }
  }

  private async load() {
    const sessionID = this.sessionID;
    if (!sessionID) return;
    this.loadState = "loading";
    this.loadError = null;
    this.emitNow();
    const epoch = this.beginEpoch();
    try {
      const detail = await getSessionDetail(this.api, sessionID, {
        limit: PAGE_SIZE,
      });
      if (!this.isCurrent(epoch)) return;
      this.expertID = detail.expertID ?? this.expertID;
      this.loadState = "ready";
      await this.adoptServerState(detail, epoch, null);
    } catch (error) {
      if (!this.isCurrent(epoch)) return;
      this.loadState = "error";
      this.loadError = describeChatError(error);
      this.emitNow();
    }
  }

  /**
   * Makes the server's view authoritative: history from its newest page,
   * then whatever turn it reports as running or queued. `finishedTurn` is a
   * turn this device just saw finish, which a lagging registry may still
   * report as active.
   */
  private async adoptServerState(
    detail: SessionDetail,
    epoch: number,
    finishedTurn: string | null,
  ) {
    this.belongsToTurn = null;
    this.live = null;
    this.replayedUserContents = [];
    this.applyNewestPage(detail);
    this.dropAcceptedLocals();
    const active = detail.activeStream;
    const sessionID = this.sessionID;
    if (sessionID && active && active.turnID !== finishedTurn) {
      const checkpoint = active.checkpoint;
      if (
        checkpoint &&
        detail.hasMore &&
        (detail.oldestSequence ?? 0) > checkpoint.sequence
      ) {
        // The newest page starts inside the turn; fetch enough rows to seed it whole.
        const wider = await getSessionDetail(this.api, sessionID, {
          limit: MAX_PAGE_SIZE,
        }).catch(() => null);
        if (!this.isCurrent(epoch)) return;
        if (wider) this.applyNewestPage(wider);
      }
      const split = splitActiveTurn(this.history, active);
      this.belongsToTurn = split.belongsToTurn;
      this.history = this.history.filter((row) => !split.belongsToTurn(row));
      this.live = split.seed;
      this.replayedUserContents = split.replayedUserContents;
      void this.runTurn(
        this.beginEpoch(),
        (signal) =>
          resumeTurnStream(this.api, sessionID, {
            turnID: active.turnID,
            after: split.after,
            signal,
          }),
        null,
      );
      return;
    }
    if (detail.chatStatus === "queued") {
      void this.waitForQueuedTurn(this.beginEpoch());
      return;
    }
    this.turn = "idle";
    this.emitNow();
  }

  private async deliver(local: LocalMessage) {
    this.turn = "sending";
    this.notice = null;
    this.emitNow();
    let sessionID = this.sessionID;
    if (!sessionID) {
      try {
        const created = await createSession(this.api, {
          expertID: this.expertID,
        });
        sessionID = created.id;
        this.sessionID = created.id;
        this.expertID = created.expertID ?? this.expertID;
        this.onSessionCreated(created.id);
      } catch (error) {
        this.failLocal(local.messageID, describeChatError(error));
        return;
      }
    }
    const target = sessionID;
    this.live = { ...emptyTurnLog(), status: "running" };
    this.replayedUserContents = [];
    await this.runTurn(
      this.beginEpoch(),
      (signal) =>
        openTurnStream(this.api, target, {
          message: local.text,
          messageID: local.messageID,
          signal,
        }),
      local.messageID,
    );
  }

  private async runTurn(epoch: number, open: Opener, sendingID: string | null) {
    let opener = open;
    let pending = sendingID;
    let failures = 0;
    this.turn = pending ? "sending" : "streaming";
    this.emitNow();
    while (this.isCurrent(epoch)) {
      const controller = new AbortController();
      this.connection = controller;
      const result = await runConnection({
        open: opener,
        controller,
        onAccepted: () => {
          if (!this.isCurrent(epoch)) return;
          if (pending) this.patchLocal(pending, { state: "accepted" });
          pending = null;
          if (!this.isStopping()) this.turn = "streaming";
          this.emitNow();
        },
        onEntry: (entry) => {
          if (!this.isCurrent(epoch)) return;
          this.live = applyEntry(
            this.live ?? { ...emptyTurnLog(), status: "running" },
            entry,
          );
          failures = 0;
          this.scheduleEmit();
        },
      });
      if (!this.isCurrent(epoch)) return;
      if (pending) {
        await this.settleUnconfirmedSend(epoch, pending, result);
        return;
      }
      const log = this.live;
      if (log && (log.status === "finished" || log.status === "failed")) {
        await this.finishTurn(epoch);
        return;
      }
      if (result.kind === "refused") {
        if ((await this.handleResumeRefusal(epoch, result.error)) === "settled")
          return;
      } else if (
        result.kind === "ended" &&
        (result.entries === 0 || !log?.turnID)
      ) {
        // Nothing (more) to stream: the send was queued or de-duplicated, or
        // the turn ended without a finish entry. Ask the server what happened.
        await this.reconcileFromServer(
          epoch,
          log?.turnID ? EMPTY_RESUME_PAUSE_MS : 0,
        );
        return;
      }
      failures += 1;
      if (failures > MAX_RECONNECTS) {
        this.turn = "offline";
        this.notice =
          "Lost the connection to this conversation. It keeps running on the server.";
        this.emitNow();
        return;
      }
      // A Stop requested while the connection was down stays "stopping".
      if (!this.isStopping()) this.turn = "reconnecting";
      this.emitNow();
      await delay(
        BACKOFF_MS[Math.min(failures, BACKOFF_MS.length) - 1],
        this.wake.signal,
      );
      if (!this.isCurrent(epoch)) return;
      const turnID = this.live?.turnID;
      const sessionID = this.sessionID;
      if (!turnID || !sessionID) {
        await this.reconcileFromServer(epoch, 0);
        return;
      }
      const after = this.live?.cursor ?? "0-0";
      opener = (signal) =>
        resumeTurnStream(this.api, sessionID, { turnID, after, signal });
    }
  }

  /** 409: the cursor fell behind a trim, so reseed from its checkpoint. 410/404: the stream is gone. */
  private async handleResumeRefusal(
    epoch: number,
    error: HttpError,
  ): Promise<"retry" | "settled"> {
    const sessionID = this.sessionID;
    const turnID = this.live?.turnID;
    const checkpoint = parseCheckpoint(
      (error.detail as { checkpoint?: unknown } | null)?.checkpoint,
    );
    if (error.status === 409 && sessionID && turnID && checkpoint) {
      try {
        const detail = await getSessionDetail(this.api, sessionID, {
          limit: MAX_PAGE_SIZE,
        });
        if (!this.isCurrent(epoch)) return "settled";
        const rows = rowsFromPersisted(detail.messages);
        const split = splitActiveTurn(rows, {
          turnID,
          startedAt: null,
          checkpoint,
        });
        this.belongsToTurn = split.belongsToTurn;
        this.history = mergeRows(
          this.withoutTurnRows(this.history),
          this.withoutTurnRows(rows),
        );
        this.live = seedTurnLog({ turnID, rows, checkpoint });
        this.replayedUserContents = [];
      } catch {
        // Retry the resume; the next refusal brings a fresh checkpoint.
      }
      return "retry";
    }
    if (error.status === 409 || error.status === 410 || error.status === 404) {
      await this.finishTurn(epoch);
      return "settled";
    }
    return "retry";
  }

  /**
   * The send's response never arrived, so the server may or may not hold the
   * message. If the newest user row is this message it was delivered;
   * otherwise it shows as failed for a manual retry, which the server
   * de-duplicates by the same idempotency key.
   */
  private async settleUnconfirmedSend(
    epoch: number,
    messageID: string,
    result: ConnectionResult,
  ) {
    if (result.kind === "refused") {
      this.live = null;
      this.failLocal(messageID, describeChatError(result.error));
      return;
    }
    const sessionID = this.sessionID;
    const local = this.locals.find((l) => l.messageID === messageID);
    if (sessionID && local) {
      try {
        const detail = await getSessionDetail(this.api, sessionID, {
          limit: PAGE_SIZE,
        });
        if (!this.isCurrent(epoch)) return;
        const newestUser = rowsFromPersisted(detail.messages)
          .filter((row) => row.role === "user")
          .at(-1);
        const delivered =
          newestUser !== undefined &&
          splitAttachments(newestUser.content).text === local.text &&
          (newestUser.createdAt === null ||
            Date.parse(newestUser.createdAt) >= local.createdAt - 60_000);
        if (delivered) {
          this.locals = this.locals.filter((l) => l.messageID !== messageID);
          await this.adoptServerState(detail, epoch, null);
          return;
        }
      } catch {
        // Still unreachable; fall through to the explicit failure.
      }
    }
    if (!this.isCurrent(epoch)) return;
    this.live = null;
    this.failLocal(
      messageID,
      "Couldn't confirm this message was sent. Retry sends it once.",
    );
  }

  private async reconcileFromServer(epoch: number, pauseMs: number) {
    const sessionID = this.sessionID;
    if (!sessionID) return;
    if (pauseMs > 0) {
      await delay(pauseMs, this.wake.signal);
      if (!this.isCurrent(epoch)) return;
    }
    try {
      const detail = await getSessionDetail(this.api, sessionID, {
        limit: PAGE_SIZE,
      });
      if (this.isCurrent(epoch))
        await this.adoptServerState(detail, epoch, null);
    } catch (error) {
      if (!this.isCurrent(epoch)) return;
      this.turn = "offline";
      this.notice = describeChatError(error);
      this.emitNow();
    }
  }

  private async waitForQueuedTurn(epoch: number) {
    this.turn = "queued";
    this.emitNow();
    for (let poll = 0; poll < QUEUED_POLLS; poll += 1) {
      await delay(QUEUED_POLL_MS, this.wake.signal);
      const sessionID = this.sessionID;
      if (!this.isCurrent(epoch) || !sessionID) return;
      const detail = await getSessionDetail(this.api, sessionID, {
        limit: PAGE_SIZE,
      }).catch(() => null);
      if (!this.isCurrent(epoch)) return;
      if (detail && (detail.activeStream || detail.chatStatus !== "queued")) {
        await this.adoptServerState(detail, epoch, null);
        return;
      }
    }
    if (!this.isCurrent(epoch)) return;
    this.turn = "idle";
    this.notice =
      "Still waiting for a free slot to start this conversation's next reply.";
    this.emitNow();
  }

  /** Replaces the live view with persisted history once the turn has ended. */
  private async finishTurn(epoch: number) {
    const sessionID = this.sessionID;
    if (!sessionID) return;
    const finishedTurn = this.live?.turnID ?? null;
    try {
      const detail = await getSessionDetail(this.api, sessionID, {
        limit: PAGE_SIZE,
      });
      if (this.isCurrent(epoch))
        await this.adoptServerState(detail, epoch, finishedTurn);
    } catch {
      if (!this.isCurrent(epoch)) return;
      // Keep the finished turn on screen; history catches up on the next refresh.
      this.turn = "idle";
      this.notice = "Couldn't refresh this conversation.";
      this.emitNow();
    }
  }

  private applyNewestPage(detail: SessionDetail) {
    const rows = this.withoutTurnRows(rowsFromPersisted(detail.messages));
    const newestOldest = detail.oldestSequence;
    // Older pages already loaded are kept; the newest page replaces its own range.
    const kept =
      newestOldest === null
        ? []
        : this.history.filter(
            (row) => row.sequence !== null && row.sequence < newestOldest,
          );
    this.history = mergeRows(kept, rows);
    if (kept.length === 0) {
      this.oldestSequence = newestOldest;
      this.hasOlder = detail.hasMore;
    }
  }

  private withoutTurnRows(rows: Row[]): Row[] {
    const belongs = this.belongsToTurn;
    return belongs ? rows.filter((row) => !belongs(row)) : rows;
  }

  private dropAcceptedLocals() {
    this.locals = this.locals.filter((l) => l.state !== "accepted");
  }

  private failLocal(messageID: string, error: string) {
    this.patchLocal(messageID, { state: "failed", error });
    this.turn = "idle";
    this.emitNow();
  }

  private patchLocal(messageID: string, patch: Partial<LocalMessage>) {
    this.locals = this.locals.map((l) =>
      l.messageID === messageID ? { ...l, ...patch } : l,
    );
  }

  private onAppStateChange(state: AppStateStatus) {
    if (state === "background") {
      this.backgroundedAt = Date.now();
      return;
    }
    if (state !== "active" || this.disposed) return;
    const away =
      this.backgroundedAt === null ? 0 : Date.now() - this.backgroundedAt;
    this.backgroundedAt = null;
    // A suspended app's socket is usually dead; resume from the cursor at once.
    if (away > FOREGROUND_RECONNECT_AFTER_MS || this.turn === "offline")
      void this.refresh();
  }

  private beginEpoch(): number {
    this.epoch += 1;
    this.connection?.abort();
    this.connection = null;
    this.rouse();
    return this.epoch;
  }

  /** Cuts short any backoff or poll wait. */
  private rouse() {
    this.wake.abort();
    this.wake = new AbortController();
  }

  private isCurrent(epoch: number): boolean {
    return !this.disposed && epoch === this.epoch;
  }

  /** Read through a method: callbacks and other calls change `turn` across awaits. */
  private isStopping(): boolean {
    return this.turn === "stopping";
  }

  private scheduleEmit() {
    if (this.emitTimer) return;
    this.emitTimer = setTimeout(() => {
      this.emitTimer = null;
      this.emitNow();
    }, EMIT_THROTTLE_MS);
  }

  private emitNow() {
    if (this.emitTimer) {
      clearTimeout(this.emitTimer);
      this.emitTimer = null;
    }
    this.snapshot = this.buildSnapshot();
    for (const listener of this.listeners) listener();
  }

  private buildSnapshot(): ConversationSnapshot {
    const running = this.live?.status === "running";
    const cache = this.historyCache;
    const persisted =
      cache && cache.history === this.history && cache.running === running
        ? cache.items
        : historyItems(this.history, running);
    this.historyCache = { history: this.history, running, items: persisted };
    return {
      sessionID: this.sessionID,
      expertID: this.expertID,
      loadState: this.loadState,
      loadError: this.loadError,
      items: [
        ...persisted,
        ...this.locals.filter((l) => l.state !== "failed").map(localItem),
        ...(this.live ? liveItems(this.live, this.replayedUserContents) : []),
        ...this.locals.filter((l) => l.state === "failed").map(localItem),
      ],
      hasOlder: this.hasOlder,
      loadingOlder: this.loadingOlder,
      refreshing: this.refreshing,
      turn: this.turn,
      statusText: running ? (this.live?.statusText ?? null) : null,
      notice: this.notice,
    };
  }
}
