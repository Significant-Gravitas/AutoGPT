import { getV2GetPendingMessages } from "@/app/api/__generated__/endpoints/chat/chat";
import type { UIMessage } from "ai";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { makePromotedUserBubble } from "./helpers/makePromotedBubble";

const MID_TURN_POLL_MS = 2_000;

type ChatStatus = "submitted" | "streaming" | "ready" | "error";

interface Chip {
  id: string;
  text: string;
}

type ChipUpdater = (prev: Chip[]) => Chip[];

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
  const [chips, setChips] = useState<Chip[]>([]);
  // Stable string view for consumers that only need the texts. Memoised so
  // downstream components don't re-render on identity churn alone.
  const queuedMessages = useMemo(() => chips.map((c) => c.text), [chips]);

  usePeekOnBoundary({ sessionId, status, setChips });

  useAutoContinuePromotion({
    sessionId,
    status,
    messages,
    chips,
    setMessages,
    setChips,
  });

  useMidTurnDrainPromotion({
    sessionId,
    status,
    chips,
    setMessages,
    setChips,
  });

  const appendChip = useCallback((text: string) => {
    setChips((prev) => [...prev, { id: crypto.randomUUID(), text }]);
  }, []);

  return { queuedMessages, appendChip };
}

// ── 1. Peek sync ───────────────────────────────────────────────────────
// Restore chips from Redis on session load + any time a turn ends (the
// backend may have drained; we reconcile with server truth).  Also
// re-peeks on `submitted → streaming` so turn-start drains reconcile
// without a separate effect — one edge-triggered peek covers both cases.

function usePeekOnBoundary({
  sessionId,
  status,
  setChips,
}: {
  sessionId: string | null;
  status: ChatStatus;
  setChips: (updater: ChipUpdater) => void;
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
    if (sessionChanged) setChips(() => []);

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
    // Capture the id-set of chips currently in local state so the
    // resolve handler can preserve any chip the user queues during the
    // GET window.
    setChips((current) => {
      inFlightSnapshotIdsRef.current = new Set(current.map((c) => c.id));
      return current;
    });
    void getV2GetPendingMessages(sessionId).then((res) => {
      if (prevSessionIdRef.current !== requestSessionId) return;
      if (res.status !== 200) return;
      const inFlightIds = inFlightSnapshotIdsRef.current;
      // Turn-start drain path: only clear chips that were in-flight at
      // GET time when the backend really emptied the buffer.  Chips
      // appended after the GET fired (during the React render between
      // turn-start and the GET resolving) survive.
      if (turnStarting && !sessionChanged) {
        if (res.data.count === 0) {
          setChips((current) => current.filter((c) => !inFlightIds.has(c.id)));
        }
        return;
      }
      // Session-load or idle-after-turn: rebase to server truth, then
      // re-attach any chips the user queued during the GET window
      // (they're not in inFlightIds because the snapshot was taken at
      // GET fire time).  Without this re-attach, a chip queued after
      // an "idle" transition but before the peek resolves silently
      // disappears.
      setChips((current) => {
        const fromServer = res.data.messages.map((text) => ({
          id: crypto.randomUUID(),
          text,
        }));
        const queuedDuringWindow = current.filter(
          (c) => !inFlightIds.has(c.id),
        );
        return [...fromServer, ...queuedDuringWindow];
      });
    });
  }, [sessionId, status, setChips]);
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

function useAutoContinuePromotion({
  sessionId,
  status,
  messages,
  chips,
  setMessages,
  setChips,
}: {
  sessionId: string | null;
  status: ChatStatus;
  messages: UIMessage[];
  chips: Chip[];
  setMessages: (updater: (prev: UIMessage[]) => UIMessage[]) => void;
  setChips: (updater: ChipUpdater) => void;
}) {
  const prevStatusRef = useRef(status);
  // The opener is the first assistant id observed after a turn starts.
  // Any LATER assistant id in the same chain is the auto-continue.
  // Reset to null on every turn boundary (turn-start or becameIdle) so
  // the next chain starts fresh.
  const openerAssistantIdRef = useRef<string | null>(null);

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
    // A different id means the backend auto-continued.
    if (chips.length === 0) return;

    const promotedIds = new Set(chips.map((c) => c.id));
    promoteBeforeAssistant(setMessages, latest, chips);
    // Drop only the chips we promoted; chips appended after the snapshot
    // (during the React commit) survive.
    setChips((current) => current.filter((c) => !promotedIds.has(c.id)));
  }, [messages, status, sessionId, chips, setMessages, setChips]);
}

// One stable bubble id per chip, shared between auto-continue and mid-turn
// promotion paths.  Without this, a poll resolving after auto-continue
// already promoted the same chip would render the chip twice (different
// ids → dedup misses).
function bubbleIdFor(chip: Chip): string {
  return `pending-chip-${chip.id}`;
}

function promoteBeforeAssistant(
  setMessages: (updater: (prev: UIMessage[]) => UIMessage[]) => void,
  assistantId: string,
  chips: Chip[],
): void {
  setMessages((prev) => {
    const idx = prev.findIndex((m) => m.id === assistantId);
    const insertAt = idx === -1 ? prev.length : idx;
    const newBubbles = chips
      .map((chip) =>
        makePromotedUserBubble(chip.text, "auto-continue", bubbleIdFor(chip)),
      )
      // Skip bubbles that are already in the array (effect re-run safety).
      .filter((bubble) => !prev.some((m) => m.id === bubble.id));
    if (newBubbles.length === 0) return prev;
    return [...prev.slice(0, insertAt), ...newBubbles, ...prev.slice(insertAt)];
  });
}

// ── 3. Mid-turn drain promotion ────────────────────────────────────────
// The MCP tool wrapper can drain the buffer at a tool boundary without
// emitting an SSE event, so the client doesn't know until we poll. On
// every poll, if the backend count dropped below our local chip count,
// promote the difference and keep the remainder as chips.
//
// TODO(followup): replace the 2s poll with an SSE event pushed from the
// backend at drain time — the MCP wrapper already knows when it drains,
// so a single "pending:drained" event would let us drop this effect
// entirely.  Tracked separately from this PR.

function useMidTurnDrainPromotion({
  sessionId,
  status,
  chips,
  setMessages,
  setChips,
}: {
  sessionId: string | null;
  status: ChatStatus;
  chips: Chip[];
  setMessages: (updater: (prev: UIMessage[]) => UIMessage[]) => void;
  setChips: (updater: ChipUpdater) => void;
}) {
  // Live ref tracks the latest sessionId so a poll captured at request
  // time can detect a session switch on resolve.  A cancellation flag
  // would also fire on chip-append (this effect re-runs on every chip
  // change), wrongly aborting an in-flight poll for the same session —
  // the sessionId comparison only invalidates on actual session changes.
  const latestSessionIdRef = useRef<string | null>(sessionId);
  useEffect(() => {
    latestSessionIdRef.current = sessionId;
  }, [sessionId]);

  useEffect(() => {
    if (!sessionId) return;
    const isActive = status === "streaming" || status === "submitted";
    if (!isActive || chips.length === 0) return;

    const requestSessionId = sessionId;
    const isCurrentSession = () =>
      latestSessionIdRef.current === requestSessionId;
    const interval = setInterval(() => {
      void pollBackendAndPromote(
        sessionId,
        chips,
        setMessages,
        setChips,
        isCurrentSession,
      );
    }, MID_TURN_POLL_MS);
    return () => clearInterval(interval);
  }, [sessionId, status, chips, setMessages, setChips]);
}

async function pollBackendAndPromote(
  sessionId: string,
  snapshotChips: Chip[],
  setMessages: (updater: (prev: UIMessage[]) => UIMessage[]) => void,
  setChips: (updater: ChipUpdater) => void,
  isCurrentSession: () => boolean,
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
  // promoting these chips to messages for a different session would
  // leak old-session bubbles into the new session.
  if (!isCurrentSession()) return;
  if (snapshotChips.length === 0) return;
  if (backendCount >= snapshotChips.length) return;

  const drainedCount = snapshotChips.length - backendCount;
  const drained = snapshotChips.slice(0, drainedCount);
  const drainedIds = new Set(drained.map((c) => c.id));

  // Splice the promoted bubble at ``len-1`` so the trailing streaming
  // assistant stays at ``messages[-1]``.  AI SDK's ``useChat`` streams
  // every SSE text/tool delta into the last message; pushing the user
  // bubble onto the tail makes ``[-1]`` the user bubble and every
  // subsequent chunk lands in the wrong slot (silently) until a page
  // refresh.  Inserting before the assistant keeps the stream flowing.
  //
  // The one tradeoff: during streaming the promoted bubbles cluster
  // just above the current streaming assistant — which is earlier in
  // the chronological order than the DB-canonical spot (between the
  // tool result they rode in on and the continuing assistant).  AI SDK's
  // single-message-per-turn model can't represent that mid-turn split
  // client-side.  ``useHydrateOnStreamEnd`` replaces the in-memory
  // messages with the DB-canonical order once the stream ends, so the
  // bubbles snap to the correct position.
  setMessages((prev) => {
    const newBubbles = drained
      .map((chip) =>
        makePromotedUserBubble(chip.text, "midturn", bubbleIdFor(chip)),
      )
      // Skip bubbles that are already there (effect re-run safety).
      .filter((bubble) => !prev.some((m) => m.id === bubble.id));
    if (newBubbles.length === 0) return prev;
    const lastIdx = prev.length - 1;
    if (lastIdx >= 0 && prev[lastIdx].role === "assistant") {
      return [...prev.slice(0, lastIdx), ...newBubbles, prev[lastIdx]];
    }
    return [...prev, ...newBubbles];
  });
  // Drop only the drained chips by id; chips appended after the snapshot
  // survive the in-flight poll race.
  setChips((current) => current.filter((c) => !drainedIds.has(c.id)));
}
