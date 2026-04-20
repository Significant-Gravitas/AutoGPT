import { getV2GetPendingMessages } from "@/app/api/__generated__/endpoints/chat/chat";
import type { UIMessage } from "ai";
import { useCallback, useEffect, useRef, useState } from "react";

import { makePromotedUserBubble } from "./helpers/makePromotedBubble";

const MID_TURN_POLL_MS = 2_000;

type ChatStatus = "submitted" | "streaming" | "ready" | "error";

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
 * State machine:
 *
 *   ┌────────┐  user queues  ┌────────────┐  backend turn-start drain
 *   │ empty  │ ───────────▶  │  showing   │ ─────────────────────────┐
 *   └────────┘               │   chips    │                          │
 *        ▲                   └────────────┘                          │
 *        │                                                           │
 *        │            ┌──────────────────────────────────────────────┘
 *        │            │ 1. auto-continue chain: promote combined bubble
 *        │            │ 2. mid-turn poll sees count drop: promote partial
 *        └────────────┘ 3. stream ends, hydration takes over
 */
export function useCopilotPendingChips({
  sessionId,
  status,
  messages,
  setMessages,
}: Args) {
  const [queuedMessages, setQueuedMessages] = useState<string[]>([]);

  usePeekOnBoundary({ sessionId, status, setQueuedMessages });

  useAutoContinuePromotion({
    sessionId,
    status,
    messages,
    queuedMessages,
    setMessages,
    setQueuedMessages,
  });

  useMidTurnDrainPromotion({
    sessionId,
    status,
    queuedMessages,
    setMessages,
    setQueuedMessages,
  });

  const appendChip = useCallback((text: string) => {
    setQueuedMessages((prev) => [...prev, text]);
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
  setQueuedMessages,
}: {
  sessionId: string | null;
  status: ChatStatus;
  setQueuedMessages: (v: string[]) => void;
}) {
  const prevSessionIdRef = useRef<string | null>(sessionId);
  const prevStatusRef = useRef<ChatStatus>(status);

  useEffect(() => {
    const prevStatus = prevStatusRef.current;
    const sessionChanged = prevSessionIdRef.current !== sessionId;
    prevSessionIdRef.current = sessionId;
    prevStatusRef.current = status;

    // Clear any stale chips from the previous session before the peek
    // resolves — otherwise the new session briefly shows the old session's
    // chips against its own messages.
    if (sessionChanged) setQueuedMessages([]);

    if (!sessionId) return;

    const isIdle = status === "ready" || status === "error";
    const turnStarting = prevStatus === "submitted" && status === "streaming";

    // Peek on: session-change, idle (covers both first-mount-in-idle and
    // becameIdle transitions), and turn-start drain.  One effect, three
    // edges — replaces the previous split between usePeekSync and the
    // auto-continue effect's duplicate turn-start peek.
    if (!sessionChanged && !isIdle && !turnStarting) return;

    void getV2GetPendingMessages(sessionId).then((res) => {
      if (res.status !== 200) return;
      // Turn-start drain path: only clear if the backend really emptied
      // the buffer.  A non-zero count means our chips survived the drain
      // (e.g. the turn is still consuming them mid-round) — keep them.
      if (turnStarting && !sessionChanged) {
        if (res.data.count === 0) setQueuedMessages([]);
        return;
      }
      // Session-load or idle-after-turn: replace with server truth.
      setQueuedMessages(res.data.count > 0 ? res.data.messages : []);
    });
  }, [sessionId, status, setQueuedMessages]);
}

// ── 2. Auto-continue promotion ─────────────────────────────────────────
// When the backend auto-continues (a SECOND new assistant ID appears in
// the same stream chain), combine chips into one user bubble and insert
// it just before that assistant — matching the DB's chronological order.
//
// Tracking model: remember the FIRST assistant id seen after
// `submitted → streaming` (that's Turn 1's opener).  Any later new
// assistant id in the same chain is the auto-continue.  Reset on every
// turn boundary.

function useAutoContinuePromotion({
  sessionId,
  status,
  messages,
  queuedMessages,
  setMessages,
  setQueuedMessages,
}: {
  sessionId: string | null;
  status: ChatStatus;
  messages: UIMessage[];
  queuedMessages: string[];
  setMessages: (updater: (prev: UIMessage[]) => UIMessage[]) => void;
  setQueuedMessages: (v: string[]) => void;
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
    if (queuedMessages.length === 0) return;

    promoteBeforeAssistant(setMessages, latest, queuedMessages);
    setQueuedMessages([]);
  }, [
    messages,
    status,
    sessionId,
    queuedMessages,
    setMessages,
    setQueuedMessages,
  ]);
}

function promoteBeforeAssistant(
  setMessages: (updater: (prev: UIMessage[]) => UIMessage[]) => void,
  assistantId: string,
  texts: string[],
): void {
  setMessages((prev) => {
    const bubbleId = `promoted-auto-continue-${assistantId}`;
    if (prev.some((m) => m.id === bubbleId)) return prev;
    const bubble = makePromotedUserBubble(texts, "auto-continue", assistantId);
    const idx = prev.findIndex((m) => m.id === assistantId);
    const insertAt = idx === -1 ? prev.length : idx;
    return [...prev.slice(0, insertAt), bubble, ...prev.slice(insertAt)];
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
  queuedMessages,
  setMessages,
  setQueuedMessages,
}: {
  sessionId: string | null;
  status: ChatStatus;
  queuedMessages: string[];
  setMessages: (updater: (prev: UIMessage[]) => UIMessage[]) => void;
  setQueuedMessages: (v: string[]) => void;
}) {
  useEffect(() => {
    if (!sessionId) return;
    const isActive = status === "streaming" || status === "submitted";
    if (!isActive || queuedMessages.length === 0) return;

    const interval = setInterval(() => {
      void pollBackendAndPromote(
        sessionId,
        queuedMessages,
        setMessages,
        setQueuedMessages,
      );
    }, MID_TURN_POLL_MS);
    return () => clearInterval(interval);
  }, [sessionId, status, queuedMessages, setMessages, setQueuedMessages]);
}

async function pollBackendAndPromote(
  sessionId: string,
  localChips: string[],
  setMessages: (updater: (prev: UIMessage[]) => UIMessage[]) => void,
  setQueuedMessages: (v: string[]) => void,
): Promise<void> {
  let backendCount: number;
  try {
    const res = await getV2GetPendingMessages(sessionId);
    if (res.status !== 200) return;
    backendCount = res.data.count;
  } catch {
    return; // harmless; next tick or hydration will reconcile
  }
  if (localChips.length === 0) return;
  if (backendCount >= localChips.length) return;

  const drainedCount = localChips.length - backendCount;
  const drained = localChips.slice(0, drainedCount);
  const remaining = localChips.slice(drainedCount);

  // Splice the promoted bubble at ``len-1`` so the trailing streaming
  // assistant stays at ``messages[-1]``.  AI SDK's ``useChat`` streams
  // every SSE text/tool delta into the last message; pushing the user
  // bubble onto the tail makes ``[-1]`` the user bubble and every
  // subsequent chunk lands in the wrong slot (silently) until a page
  // refresh.  Inserting before the assistant keeps the stream flowing.
  //
  // The one tradeoff: during streaming the promoted bubble clusters
  // just above the current streaming assistant — which is earlier in
  // the chronological order than the DB-canonical spot (between the
  // tool result it rode in on and the continuing assistant).  AI SDK's
  // single-message-per-turn model can't represent that mid-turn split
  // client-side.  ``useHydrateOnStreamEnd`` replaces the in-memory
  // messages with the DB-canonical order once the stream ends, so the
  // bubble snaps to the correct position.
  setMessages((prev) => {
    const bubble = makePromotedUserBubble(
      drained,
      "midturn",
      crypto.randomUUID(),
    );
    const lastIdx = prev.length - 1;
    if (lastIdx >= 0 && prev[lastIdx].role === "assistant") {
      return [...prev.slice(0, lastIdx), bubble, prev[lastIdx]];
    }
    return [...prev, bubble];
  });
  setQueuedMessages(remaining);
}
