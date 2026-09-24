import { getV2GetPendingMessages } from "@/app/api/__generated__/endpoints/chat/chat";
import type { UIMessage } from "ai";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { makePromotedUserBubble } from "./helpers/makePromotedBubble";
import { v4 as uuidv4 } from "uuid";
import { PENDING_DRAINED_PART_TYPE } from "./messageParts";

// Backstop only. Promotion is normally driven instantly by the backend's
// ``data-pending-drained`` SSE hint (see ``useMidTurnDrainPromotion``); this
// slow poll just catches a dropped hint so a chip can't get stuck mid-turn.
const MID_TURN_BACKSTOP_POLL_MS = 10_000;

type ChatStatus = "submitted" | "streaming" | "ready" | "error";

interface QueuedMessage {
  id: string;
  text: string;
  /** Restored from the backend buffer by a peek rather than typed here.
   *  The next peek's rebase replaces these; only entries typed during its
   *  GET window are carried over. */
  fromServer?: boolean;
}

type QueueUpdater = (prev: QueuedMessage[]) => QueuedMessage[];

interface Args {
  sessionId: string | null;
  status: ChatStatus;
  messages: UIMessage[];
  setMessages: (
    updater: UIMessage[] | ((prev: UIMessage[]) => UIMessage[]),
  ) => void;
}

/**
 * Owns the chip lifecycle: keep the local chip list in sync with Redis,
 * promote chips to user-bubbles when the backend drains (auto-continue or
 * mid-turn via the MCP wrapper), and surface the list + queue op to the
 * chat input.
 *
 * Each chip carries a frontend-only ``id`` so concurrent draining and
 * appending stays race-safe — the poll captures a stale snapshot of the
 * chip array, but the eventual state mutation drops chips by id rather
 * than by array slice, so a chip the user enqueues during the in-flight
 * poll cannot be silently overwritten.
 *
 * State machine:
 *
 *   ┌────────┐  user queues  ┌────────────┐  backend turn-start drain
 *   │ empty  │ ───────────▶  │  showing   │ ─────────────────────────┐
 *   └────────┘               │   chips    │                          │
 *        ▲                   └────────────┘                          │
 *        │                                                           │
 *        │            ┌──────────────────────────────────────────────┘
 *        │            │ 1. auto-continue chain: promote one bubble per chip
 *        │            │ 2. mid-turn poll sees count drop: promote drained chips
 *        └────────────┘ 3. stream ends, hydration takes over
 */
export function useCopilotPendingChips({
  sessionId,
  status,
  messages,
  setMessages,
}: Args) {
  const [queue, setQueue] = useState<QueuedMessage[]>([]);
  // Stable string view for consumers that only need the texts. Memoised so
  // downstream components don't re-render on identity churn alone.
  const queuedMessages = useMemo(
    () => queue.map((entry) => entry.text),
    [queue],
  );

  usePeekOnBoundary({ sessionId, status, setMessages, setQueue });

  useAutoContinuePromotion({
    sessionId,
    status,
    messages,
    queue,
    setMessages,
    setQueue,
  });

  useMidTurnDrainPromotion({
    sessionId,
    status,
    messages,
    queue,
    setMessages,
    setQueue,
  });

  const queueMessage = useCallback((text: string) => {
    // Options force uuid's getRandomValues path: crypto.randomUUID does not
    // exist on a plain-HTTP LAN origin, and this updater runs during render,
    // so there it took the whole chat page down instead of queueing.
    setQueue((prev) => [...prev, { id: uuidv4({}), text }]);
  }, []);

  return { queuedMessages, queueMessage };
}

// ── 1. Peek sync ───────────────────────────────────────────────────────
// Restore chips from Redis on session load + any time a turn ends (the
// backend may have drained; we reconcile with server truth).  Also
// re-peeks on `submitted → streaming` so turn-start drains reconcile
// without a separate effect — one edge-triggered peek covers both cases.

function usePeekOnBoundary({
  sessionId,
  status,
  setMessages,
  setQueue,
}: {
  sessionId: string | null;
  status: ChatStatus;
  setMessages: (updater: (prev: UIMessage[]) => UIMessage[]) => void;
  setQueue: (updater: QueueUpdater) => void;
}) {
  const prevSessionIdRef = useRef<string | null>(sessionId);
  const prevStatusRef = useRef<ChatStatus>(status);
  // Snapshot of chip ids known to be in-flight to the server at the
  // moment a peek GET is issued.  Anything NOT in this set when the GET
  // resolves was appended after the request — preserve it so a
  // concurrently-queued chip isn't wiped by the server's now-stale
  // truth.  Set inside the effect (closes over the current chips
  // value), read inside the ``.then`` handler.
  const inFlightSnapshotIdsRef = useRef<Set<string>>(new Set());

  useEffect(() => {
    const prevStatus = prevStatusRef.current;
    const sessionChanged = prevSessionIdRef.current !== sessionId;
    prevSessionIdRef.current = sessionId;
    prevStatusRef.current = status;

    // Clear any stale chips from the previous session before the peek
    // resolves — otherwise the new session briefly shows the old session's
    // chips against its own messages.
    if (sessionChanged) setQueue(() => []);

    if (!sessionId) return;

    const isIdle = status === "ready" || status === "error";
    const turnStarting = prevStatus === "submitted" && status === "streaming";

    // Peek on: session-change, idle (covers both first-mount-in-idle and
    // becameIdle transitions), and turn-start drain.  One effect, three
    // edges — replaces the previous split between usePeekSync and the
    // auto-continue effect's duplicate turn-start peek.
    if (!sessionChanged && !isIdle && !turnStarting) return;

    // Capture sessionId at request time and compare to the live ref on
    // resolve.  Without this, a peek that resolves after the user
    // switched sessions could bleed old-session chips into the new
    // session.  We deliberately don't use a per-effect cancelled flag
    // here because this effect re-runs on every status change too, and
    // we don't want chip-appends in another effect to invalidate this
    // peek's result.
    const requestSessionId = sessionId;
    // Capture the id-set of queue entries currently in local state so
    // the resolve handler can preserve any entry the user queues during
    // the GET window.
    setQueue((current) => {
      inFlightSnapshotIdsRef.current = new Set(
        current.map((entry) => entry.id),
      );
      return current;
    });
    void getV2GetPendingMessages(sessionId).then((res) => {
      if (prevSessionIdRef.current !== requestSessionId) return;
      if (res.status !== 200) return;
      const inFlightIds = inFlightSnapshotIdsRef.current;
      // Turn-start drain path: when the backend has drained everything
      // it had at GET time, promote those drained entries to user
      // bubbles BEFORE removing them from local state.  Without the
      // promote step, the chips disappear from the chip-strip but the
      // bubble for each drained entry only shows up later via
      // hydration once the turn fully ends — leaving the user staring
      // at a streaming assistant that's responding to text they can no
      // longer see typed in the chat.  Entries appended after the GET
      // fired (during the React render between turn-start and the GET
      // resolving) survive untouched.
      if (turnStarting && !sessionChanged) {
        if (res.data.count === 0) {
          setQueue((current) => {
            const drained = current.filter((entry) =>
              inFlightIds.has(entry.id),
            );
            if (drained.length > 0) {
              promoteChipsToTrailingBubbles(setMessages, drained);
            }
            return current.filter((entry) => !inFlightIds.has(entry.id));
          });
        }
        return;
      }
      // Session-load or idle-after-turn: rebase to server truth, then
      // re-attach any entries the user queued during the GET window
      // (they're not in inFlightIds because the snapshot was taken at
      // GET fire time).  Without this re-attach, an entry queued after
      // an "idle" transition but before the peek resolves silently
      // disappears.
      //
      // Entries an *earlier* peek restored are not "queued during the
      // window" even when they post-date this GET's snapshot: two peeks
      // overlap on load (Strict Mode mounts the effect twice; two idle
      // edges can land within one round trip), and each carrying the
      // other's copy forward doubled the strip. The next new-assistant
      // reconciliation then saw more chips than the backend held and
      // promoted the surplus above the running tool chain.
      setQueue((current) => {
        const fromServer = res.data.messages.map((text) => ({
          id: uuidv4({}),
          text,
          fromServer: true,
        }));
        const queuedDuringWindow = current.filter(
          (entry) => !inFlightIds.has(entry.id) && !entry.fromServer,
        );
        return [...fromServer, ...queuedDuringWindow];
      });
    });
  }, [sessionId, status, setQueue]);
}

// ── 2. Auto-continue promotion ─────────────────────────────────────────
// When the backend auto-continues (a SECOND new assistant ID appears in
// the same stream chain), promote drained chips into user bubbles before
// that assistant — matching the DB's chronological order.
//
// Tracking model: remember the FIRST assistant id seen after
// `submitted → streaming` (that's Turn 1's opener).  Any later new
// assistant id in the same chain is the auto-continue.  Reset on every
// turn boundary.
//
// A new id is only a *cue* to reconcile, never proof of a drain: the
// backend emits `data-status` before `start`, so `useChat` parks the turn
// in a placeholder under its own id and then pushes the real message once
// `start` carries the server's id — the same "new assistant id" shape, with
// the follow-up still sitting in the buffer. Promoting on the cue alone
// drew the chip as a plain user bubble above the live tool chain (twice
// after a mid-turn reload, once per peek). The buffer re-read decides.

function useAutoContinuePromotion({
  sessionId,
  status,
  messages,
  queue,
  setMessages,
  setQueue,
}: {
  sessionId: string | null;
  status: ChatStatus;
  messages: UIMessage[];
  queue: QueuedMessage[];
  setMessages: (updater: (prev: UIMessage[]) => UIMessage[]) => void;
  setQueue: (updater: QueueUpdater) => void;
}) {
  const prevStatusRef = useRef(status);
  // The opener is the first assistant id observed after a turn starts.
  // Any LATER assistant id in the same chain is the auto-continue.
  // Reset to null on every turn boundary (turn-start or becameIdle) so
  // the next chain starts fresh.
  const openerAssistantIdRef = useRef<string | null>(null);
  const latestSessionIdRef = useRef<string | null>(sessionId);
  useEffect(() => {
    latestSessionIdRef.current = sessionId;
  }, [sessionId]);

  useEffect(() => {
    const prevStatus = prevStatusRef.current;
    prevStatusRef.current = status;

    const turnStarting = prevStatus === "submitted" && status === "streaming";
    const becameIdle =
      (prevStatus === "streaming" || prevStatus === "submitted") &&
      (status === "ready" || status === "error");
    if (turnStarting || becameIdle) {
      openerAssistantIdRef.current = null;
    }

    if (!sessionId) return;
    const isActive = status === "streaming" || status === "submitted";
    if (!isActive) return;

    const assistantIds = messages
      .filter((m) => m.role === "assistant")
      .map((m) => m.id);
    if (assistantIds.length === 0) return;

    const latest = assistantIds[assistantIds.length - 1];
    // First assistant id of this chain — it's Turn 1's opener.
    if (openerAssistantIdRef.current === null) {
      openerAssistantIdRef.current = latest;
      return;
    }
    // Same id as opener — no new assistant yet, wait.
    if (latest === openerAssistantIdRef.current) return;
    // A different id: reconcile once against the buffer, then treat this
    // id as the current opener so later deltas into it stay quiet.
    openerAssistantIdRef.current = latest;
    if (queue.length === 0) return;

    const requestSessionId = sessionId;
    const isCurrentSession = () =>
      latestSessionIdRef.current === requestSessionId;
    void pollBackendAndPromote(
      sessionId,
      queue,
      setMessages,
      setQueue,
      isCurrentSession,
      "auto-continue",
    );
  }, [messages, status, sessionId, queue, setMessages, setQueue]);
}

type PromotionFlavour = Parameters<typeof makePromotedUserBubble>[1];

const PROMOTION_FLAVOURS: PromotionFlavour[] = ["auto-continue", "midturn"];

/**
 * Splice promoted user bubbles for *drained* in just before the trailing
 * streaming assistant message, so AI SDK's streaming continues into the
 * right slot.  Every promotion path funnels through here: the turn-start
 * drain in ``usePeekOnBoundary`` (the backend drained chips before the
 * first peek resolved, and the bubble would otherwise only appear via
 * hydration after the turn ends), the auto-continue reconciliation, and
 * the mid-turn hint / backstop poll.
 *
 * An entry that already has a bubble under *either* flavour is skipped.
 * Two reconciliations of the same chip can be in flight at once — a new
 * assistant id and a drain hint (or the backstop) each issue their own
 * GET — and both see the drained buffer; keyed on the exact id, each
 * flavour would store its own copy and the transcript would draw the
 * follow-up twice.
 */
function promoteChipsToTrailingBubbles(
  setMessages: (updater: (prev: UIMessage[]) => UIMessage[]) => void,
  drained: QueuedMessage[],
  flavour: PromotionFlavour = "midturn",
): void {
  setMessages((prev) => {
    const newBubbles = drained
      .filter((entry) => !prev.some((m) => isPromotedBubbleFor(m, entry)))
      .map((entry) =>
        makePromotedUserBubble(entry.text, flavour, bubbleIdFor(entry)),
      );
    if (newBubbles.length === 0) return prev;
    const lastIdx = prev.length - 1;
    if (lastIdx >= 0 && prev[lastIdx].role === "assistant") {
      return [...prev.slice(0, lastIdx), ...newBubbles, prev[lastIdx]];
    }
    return [...prev, ...newBubbles];
  });
}

// One stable bubble id per queued message, shared between auto-continue
// and mid-turn promotion paths.  Without this, a poll resolving after
// auto-continue already promoted the same entry would render the bubble
// twice (different ids → dedup misses).
function bubbleIdFor(entry: QueuedMessage): string {
  return `pending-chip-${entry.id}`;
}

// Whether *message* is the promoted bubble for *entry*, whichever path
// promoted it.  The queue id is the chip's identity; the flavour prefix
// only records which path got there first.  Repeated user messages carry
// distinct queue ids, so they still each get a bubble.
function isPromotedBubbleFor(
  message: UIMessage,
  entry: QueuedMessage,
): boolean {
  const suffix = bubbleIdFor(entry);
  return PROMOTION_FLAVOURS.some(
    (flavour) =>
      message.id === makePromotedUserBubble(entry.text, flavour, suffix).id,
  );
}

// ── 3. Mid-turn drain promotion ────────────────────────────────────────
// The executor drains the buffer at a tool boundary while the turn is
// still streaming.  It now pushes a ``data-pending-drained`` SSE hint at
// drain time, so the fast path promotes chips the instant the hint lands.
// A slow backstop poll covers a dropped hint so a chip can't get stuck.
//
// Both paths funnel through the same ``pollBackendAndPromote`` (the GET
// stays the source of truth: it re-reads the authoritative buffer count
// and promotes only the difference).  Promotion dedupes bubbles by a
// stable id, so the hint and the poll firing for the same drain is a
// harmless no-op rather than a double-render.

function useMidTurnDrainPromotion({
  sessionId,
  status,
  messages,
  queue,
  setMessages,
  setQueue,
}: {
  sessionId: string | null;
  status: ChatStatus;
  messages: UIMessage[];
  queue: QueuedMessage[];
  setMessages: (updater: (prev: UIMessage[]) => UIMessage[]) => void;
  setQueue: (updater: QueueUpdater) => void;
}) {
  // Live ref tracks the latest sessionId so a poll captured at request
  // time can detect a session switch on resolve.  A cancellation flag
  // would also fire on enqueue (this effect re-runs on every queue
  // change), wrongly aborting an in-flight poll for the same session —
  // the sessionId comparison only invalidates on actual session changes.
  const latestSessionIdRef = useRef<string | null>(sessionId);
  useEffect(() => {
    latestSessionIdRef.current = sessionId;
  }, [sessionId]);

  // Fast path: promote the moment the backend signals a drain.  We count
  // ``data-pending-drained`` parts across messages and react to the count
  // increasing — replays (AI SDK resume re-emits the parts) leave the
  // count unchanged on a stable render, and the GET re-read keeps a
  // replayed hint idempotent regardless.
  const drainHintCount = countPendingDrainedHints(messages);
  // Baseline is tracked per session: on a session switch the old session's
  // chips can still be in `queue` for the current commit, so a higher hint
  // count in the new session must not promote stale chips into the new chat.
  // Re-baseline (and bail) when sessionId changes before comparing counts.
  const prevHintStateRef = useRef({ sessionId, count: drainHintCount });
  useEffect(() => {
    if (prevHintStateRef.current.sessionId !== sessionId) {
      prevHintStateRef.current = { sessionId, count: drainHintCount };
      return;
    }

    const isActive = status === "streaming" || status === "submitted";
    if (!sessionId || !isActive || queue.length === 0) {
      prevHintStateRef.current = { sessionId, count: drainHintCount };
      return;
    }
    if (drainHintCount <= prevHintStateRef.current.count) return;
    prevHintStateRef.current = { sessionId, count: drainHintCount };

    const requestSessionId = sessionId;
    const isCurrentSession = () =>
      latestSessionIdRef.current === requestSessionId;
    void pollBackendAndPromote(
      sessionId,
      queue,
      setMessages,
      setQueue,
      isCurrentSession,
    );
  }, [drainHintCount, sessionId, status, queue, setMessages, setQueue]);

  // Backstop: a slow poll that catches a dropped hint.
  useEffect(() => {
    if (!sessionId) return;
    const isActive = status === "streaming" || status === "submitted";
    if (!isActive || queue.length === 0) return;

    const requestSessionId = sessionId;
    const isCurrentSession = () =>
      latestSessionIdRef.current === requestSessionId;
    const interval = setInterval(() => {
      void pollBackendAndPromote(
        sessionId,
        queue,
        setMessages,
        setQueue,
        isCurrentSession,
      );
    }, MID_TURN_BACKSTOP_POLL_MS);
    return () => clearInterval(interval);
  }, [sessionId, status, queue, setMessages, setQueue]);
}

// Count ``data-pending-drained`` hint parts the backend emits at each
// mid-turn drain.  A rising count across renders means a fresh drain the
// fast path should react to.
function countPendingDrainedHints(messages: UIMessage[]): number {
  let count = 0;
  for (const message of messages) {
    for (const part of message.parts) {
      if (part.type === PENDING_DRAINED_PART_TYPE) count++;
    }
  }
  return count;
}

async function pollBackendAndPromote(
  sessionId: string,
  snapshotQueue: QueuedMessage[],
  setMessages: (updater: (prev: UIMessage[]) => UIMessage[]) => void,
  setQueue: (updater: QueueUpdater) => void,
  isCurrentSession: () => boolean,
  flavour: PromotionFlavour = "midturn",
): Promise<void> {
  let backendCount: number;
  try {
    const res = await getV2GetPendingMessages(sessionId);
    if (res.status !== 200) return;
    backendCount = res.data.count;
  } catch {
    return; // harmless; next tick or hydration will reconcile
  }
  // Bail if the user switched sessions while the GET was in flight —
  // promoting these entries to messages for a different session would
  // leak old-session bubbles into the new session.
  if (!isCurrentSession()) return;
  if (snapshotQueue.length === 0) return;
  if (backendCount >= snapshotQueue.length) return;

  const drainedCount = snapshotQueue.length - backendCount;
  const drained = snapshotQueue.slice(0, drainedCount);
  const drainedIds = new Set(drained.map((entry) => entry.id));

  // Every drained chip becomes a fallback bubble, whether or not the
  // ``data-pending-drained`` hint carried its text. A backend that ships the
  // text lets the transcript draw the bubble at the drain point instead
  // (``splitMessagesAtDrainHints``), and the render-time split then drops
  // the fallback row it matches — so deciding here from the transcript's
  // hints is unnecessary, and unsafe: the poll cannot tell which hint was
  // this drain's, so an earlier drain's text would suppress a later chip
  // with the same words (count-only hint, dropped hint) and lose its bubble.
  //
  // Why not simply append the bubble? ``useChat`` streams every SSE delta
  // into ``messages[-1]``; pushing the user bubble onto the tail makes
  // ``[-1]`` the user bubble and every subsequent chunk lands in the wrong
  // slot (silently) until a page refresh. Inserting before the assistant
  // keeps the stream flowing, at the cost of showing a count-only follow-up
  // above the work that preceded it until ``useHydrateOnStreamEnd`` snaps
  // the list to the DB order at the end of the turn.
  promoteChipsToTrailingBubbles(setMessages, drained, flavour);
  // Drop only the drained entries by id; entries appended after the
  // snapshot survive the in-flight poll race.
  setQueue((current) => current.filter((entry) => !drainedIds.has(entry.id)));
}
