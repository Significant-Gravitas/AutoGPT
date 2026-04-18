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

  // Mirror state into refs so async callbacks and interval ticks always
  // read the latest values without re-subscribing on every change.
  const queuedRef = useRef(queuedMessages);
  queuedRef.current = queuedMessages;

  usePeekSync({ sessionId, status, setQueuedMessages });

  useAutoContinuePromotion({
    sessionId,
    status,
    messages,
    queuedRef,
    setMessages,
    setQueuedMessages,
  });

  useMidTurnDrainPromotion({
    sessionId,
    status,
    queuedLen: queuedMessages.length,
    queuedRef,
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
// backend may have drained; we reconcile with server truth).

function usePeekSync({
  sessionId,
  status,
  setQueuedMessages,
}: {
  sessionId: string | null;
  status: ChatStatus;
  setQueuedMessages: (v: string[]) => void;
}) {
  const prevSessionIdRef = useRef<string | null>(sessionId);

  useEffect(() => {
    const sessionChanged = prevSessionIdRef.current !== sessionId;
    prevSessionIdRef.current = sessionId;

    // Clear any stale chips from the previous session before the peek
    // resolves — otherwise the new session briefly shows the old session's
    // chips against its own messages.
    if (sessionChanged) setQueuedMessages([]);

    if (!sessionId) return;
    const isIdle = status === "ready" || status === "error";
    if (!sessionChanged && !isIdle) return;

    void getV2GetPendingMessages(sessionId).then((res) => {
      setQueuedMessages(
        res.status === 200 && res.data.count > 0 ? res.data.messages : [],
      );
    });
  }, [sessionId, status, setQueuedMessages]);
}

// ── 2. Auto-continue promotion ─────────────────────────────────────────
// When the backend auto-continues (a SECOND new assistant ID appears in
// the same stream chain), combine chips into one user bubble and insert
// it just before that assistant — matching the DB's chronological order.

function useAutoContinuePromotion({
  sessionId,
  status,
  messages,
  queuedRef,
  setMessages,
  setQueuedMessages,
}: {
  sessionId: string | null;
  status: ChatStatus;
  messages: UIMessage[];
  queuedRef: React.RefObject<string[]>;
  setMessages: (updater: (prev: UIMessage[]) => UIMessage[]) => void;
  setQueuedMessages: (v: string[]) => void;
}) {
  const prevStatusRef = useRef(status);
  const seenAssistantIdsRef = useRef<Set<string>>(
    new Set(messages.filter((m) => m.role === "assistant").map((m) => m.id)),
  );
  // Tracks whether we've already credited "Turn 1's assistant" so the next
  // new assistant id in the same chain is treated as the auto-continue.
  const sawTurnOpenerRef = useRef(false);

  useEffect(() => {
    const prevStatus = prevStatusRef.current;
    prevStatusRef.current = status;

    const newAssistantIds = absorbNewAssistantIds(
      messages,
      seenAssistantIdsRef.current,
    );

    if (!sessionId) return;

    const turnStarting = prevStatus === "submitted" && status === "streaming";
    if (turnStarting) {
      sawTurnOpenerRef.current = false;
      // Turn-start drain: chips got merged into the submitted message.
      // Peek once — clear chips only if the backend really drained them.
      void getV2GetPendingMessages(sessionId).then((res) => {
        if (res.status === 200 && res.data.count === 0) setQueuedMessages([]);
      });
    }

    const becameIdle =
      (prevStatus === "streaming" || prevStatus === "submitted") &&
      (status === "ready" || status === "error");
    if (becameIdle) sawTurnOpenerRef.current = false;

    const isActive = status === "streaming" || status === "submitted";
    if (!isActive || newAssistantIds.length === 0) return;

    const autoContinueId = pickAutoContinueId(
      newAssistantIds,
      sawTurnOpenerRef,
    );
    if (autoContinueId === null) return;
    if ((queuedRef.current ?? []).length === 0) return;

    promoteBeforeAssistant(
      setMessages,
      autoContinueId,
      queuedRef.current ?? [],
    );
    setQueuedMessages([]);
  }, [messages, status, sessionId, queuedRef, setMessages, setQueuedMessages]);
}

/** Return new assistant ids never seen before, and remember them. */
function absorbNewAssistantIds(
  messages: UIMessage[],
  seen: Set<string>,
): string[] {
  const current = messages
    .filter((m) => m.role === "assistant")
    .map((m) => m.id);
  const fresh = current.filter((id) => !seen.has(id));
  current.forEach((id) => seen.add(id));
  return fresh;
}

/**
 * The first new assistant of a stream is Turn 1's opener (not an auto-
 * continue). Remember it the first time, then treat any subsequent new
 * assistant in the same chain as the auto-continue.
 */
function pickAutoContinueId(
  newIds: string[],
  sawOpenerRef: React.MutableRefObject<boolean>,
): string | null {
  if (sawOpenerRef.current) return newIds[0];
  sawOpenerRef.current = true;
  // If opener + auto-continue landed in the same render batch, drop the
  // opener and treat the next id as the auto-continue. Otherwise wait
  // for the next render.
  return newIds.length >= 2 ? newIds[1] : null;
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

function useMidTurnDrainPromotion({
  sessionId,
  status,
  queuedLen,
  queuedRef,
  setMessages,
  setQueuedMessages,
}: {
  sessionId: string | null;
  status: ChatStatus;
  queuedLen: number;
  queuedRef: React.RefObject<string[]>;
  setMessages: (updater: (prev: UIMessage[]) => UIMessage[]) => void;
  setQueuedMessages: (v: string[]) => void;
}) {
  useEffect(() => {
    if (!sessionId) return;
    const isActive = status === "streaming" || status === "submitted";
    if (!isActive || queuedLen === 0) return;

    const interval = setInterval(() => {
      void pollBackendAndPromote(
        sessionId,
        queuedRef.current ?? [],
        setMessages,
        setQueuedMessages,
      );
    }, MID_TURN_POLL_MS);
    return () => clearInterval(interval);
  }, [sessionId, status, queuedLen, queuedRef, setMessages, setQueuedMessages]);
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

  setMessages((prev) => [
    ...prev,
    makePromotedUserBubble(drained, "midturn", crypto.randomUUID()),
  ]);
  setQueuedMessages(remaining);
}
