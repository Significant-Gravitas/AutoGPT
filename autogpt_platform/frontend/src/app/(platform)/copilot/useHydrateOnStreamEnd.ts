import { toast } from "@/components/molecules/Toast/use-toast";
import type { UIMessage } from "ai";
import { useEffect, useRef } from "react";

import {
  deduplicateMessages,
  hasInProgressAssistantParts,
  resolveInterruptedMessage,
} from "./helpers";
import { extractDbSequence } from "./helpers/convertChatSessionToUiMessages";
import { CANCELLED_MARKER } from "./useCopilotStop";

const PROMOTED_BUBBLE_ID_PREFIX = "promoted-";

/**
 * The hydrated payload is only the backend's most-recent tail window (the
 * `GET /sessions` endpoint returns the latest ~50 messages). That window
 * slides forward as the conversation grows, so a blind force-replace would
 * drop every in-memory message older than the new window's oldest sequence —
 * tearing a hole between the live state and any older history the user
 * already scrolled back to (`pagedMessages`), which scroll-back can't refill
 * (it only fetches messages older than what's loaded). See SECRT-2424.
 *
 * Keep the `prev` messages whose DB sequence predates the window so the
 * force-hydrated result stays contiguous with the older history. Both arrays
 * are in ascending DB-sequence order, so the window's oldest sequence is the
 * first `-seq-N` row in `hydrated`, and the older `prev` messages are the
 * leading run before that boundary — a single slice. Streaming / idx-fallback
 * rows (no `-seq-N` id) carry no sequence and mark the end of that run; they
 * are excluded because they already live inside the refetched window.
 */
function retainOlderHistory(
  prev: UIMessage[],
  hydrated: UIMessage[],
): UIMessage[] {
  const windowStart = firstSequence(hydrated);
  if (windowStart === null) return hydrated;

  const windowStartIndex = prev.findIndex((message) => {
    const seq = extractDbSequence(message);
    return seq === null || seq >= windowStart;
  });
  const olderCount = windowStartIndex === -1 ? prev.length : windowStartIndex;
  if (olderCount === 0) return hydrated;

  return [...prev.slice(0, olderCount), ...hydrated];
}

function getMessageText(message: UIMessage): string {
  return message.parts
    .map((part) => (part.type === "text" && "text" in part ? part.text : ""))
    .join("\n\n")
    .trim();
}

/**
 * When force-hydrating after a stream ends, replace the in-memory messages
 * with the canonical DB state — but preserve any ``promoted-*`` bubbles
 * (mid-turn or auto-continue chip promotions) whose text isn't reflected in
 * the hydrated user messages yet. This covers the rare case where Claude
 * already saw the queued message (via mid-turn injection) but the persist
 * step rolled back, so the DB row never landed; without this guard, the
 * user sees their message disappear from the feed even though the LLM
 * responded to it.
 */
function preservePromotedUserBubbles(
  prev: UIMessage[],
  hydrated: UIMessage[],
): UIMessage[] {
  const promoted = prev.filter(isPromotedBubble);
  if (promoted.length === 0) return hydrated;

  const hydratedUserText = hydrated
    .filter((m) => m.role === "user")
    .map(getMessageText)
    .join("\n\n");

  const orphans = promoted.filter((bubble) => {
    const bubbleText = getMessageText(bubble);
    if (!bubbleText) return false;
    return !hydratedUserText.includes(bubbleText);
  });
  if (orphans.length === 0) return hydrated;

  return [...hydrated, ...orphans];
}

/**
 * Keep a send the backend took but has not written yet (a follow-up queued
 * behind a stopped turn's teardown, SECRT-2843) until a user row with its text
 * lands after the row it was sent behind; each row answers for one bubble.
 * A bubble whose row may lie before the refetched window is left to it.
 */
function preserveUnpersistedUserBubbles(
  prev: UIMessage[],
  hydrated: UIMessage[],
  windowStart: number | null,
): UIMessage[] {
  // With no stored row drawn yet, nothing says which sends the backend took;
  // a first send refused before it was saved is one of them.
  if (!prev.some((message) => extractDbSequence(message) !== null))
    return hydrated;
  const bubbles = localUserBubbles(prev);
  const claimed = new Set<number>();
  // A promoted bubble's row is its own, so a later bubble with the same text
  // cannot take it.
  for (const bubble of bubbles.filter((b) => isPromotedBubble(b.message))) {
    const rowIndex = findUnclaimedUserRow(hydrated, claimed, bubble);
    if (rowIndex !== -1) claimed.add(rowIndex);
  }
  const orphans: UIMessage[] = [];
  for (const bubble of bubbles.filter((b) => !isPromotedBubble(b.message))) {
    const rowIndex = findUnclaimedUserRow(hydrated, claimed, bubble);
    if (rowIndex !== -1) claimed.add(rowIndex);
    else if (windowStart === null || bubble.sentAfter >= windowStart - 1)
      orphans.push(bubble.message);
  }
  if (orphans.length === 0) return hydrated;
  return [...hydrated, ...orphans];
}

interface LocalBubble {
  message: UIMessage;
  text: string;
  /** Sequence of the last stored row before it; -1 when there is none. */
  sentAfter: number;
}

function localUserBubbles(prev: UIMessage[]): LocalBubble[] {
  const bubbles: LocalBubble[] = [];
  let sentAfter = -1;
  for (const message of prev) {
    const seq = extractDbSequence(message);
    if (seq !== null) {
      sentAfter = seq;
      continue;
    }
    const text = message.role === "user" ? getMessageText(message) : "";
    if (text) bubbles.push({ message, text, sentAfter });
  }
  return bubbles;
}

function findUnclaimedUserRow(
  hydrated: UIMessage[],
  claimed: Set<number>,
  { text, sentAfter }: LocalBubble,
): number {
  return hydrated.findIndex(
    (row, index) =>
      !claimed.has(index) &&
      row.role === "user" &&
      (extractDbSequence(row) ?? -1) > sentAfter &&
      getMessageText(row).includes(text),
  );
}

function firstSequence(messages: UIMessage[]): number | null {
  for (const message of messages) {
    const seq = extractDbSequence(message);
    if (seq !== null) return seq;
  }
  return null;
}

function isPromotedBubble(message: UIMessage): boolean {
  return (
    typeof message.id === "string" &&
    message.id.startsWith(PROMOTED_BUBBLE_ID_PREFIX)
  );
}

/**
 * A stopped turn is finalised client-side — ``useCopilotStop`` appends the
 * cancellation marker — and the backend persists nothing for it, so a blind
 * force-replace drops the partial answer and the marker with it, and the error
 * banner that marker suppresses takes the stopped card's place. Re-attach the
 * local bubble while the hydrated view still ends on the user's prompt.
 */
function preserveStoppedAssistantMessage(
  prev: UIMessage[],
  hydrated: UIMessage[],
): UIMessage[] {
  const stopped = prev[prev.length - 1];
  if (stopped?.role !== "assistant") return hydrated;
  const isStopped = stopped.parts.some(
    (part) => part.type === "text" && part.text.includes(CANCELLED_MARKER),
  );
  if (!isStopped) return hydrated;
  if (hydrated[hydrated.length - 1]?.role !== "user") return hydrated;
  return [...hydrated, stopped];
}

type ChatStatus = "submitted" | "streaming" | "ready" | "error";

// Survive remount on session re-entry: useRef would reset every time
// CopilotPage's <Component key={sessionId}/> remounts the tree, causing
// the "previous response was interrupted" toast to fire again on every
// re-entry into an affected session. Module-scoped so it persists for
// the page lifetime.
const sessionsWithInterruptedToastShown = new Set<string>();

interface Args {
  /**
   * Active session id. Used to key the "interrupted-toast already shown"
   * ledger so the warning fires at most once per session per page load.
   * ``null`` is allowed for the brief window before a session is bound;
   * the toast simply won't be deduped during that window.
   */
  sessionId: string | null;
  status: ChatStatus;
  hydratedMessages: UIMessage[] | undefined;
  isReconnectScheduled: boolean;
  /**
   * Whether the backend currently has an active SSE stream for this
   * session. Used to gate zombie-part recovery — if the backend is still
   * streaming we leave in-progress parts alone (resume will fill them
   * in); if it isn't, we finalise them so spinners stop and the user can
   * resend.
   */
  hasActiveStream: boolean;
  /**
   * True while the post-finish probe (`handleFinish` in `useCopilotStream`)
   * is still deciding whether the backend is starting a continuation turn.
   * Applying the force-hydrate in that window swaps every message id (AI-SDK
   * uuid → DB `-seq-N`), remounting the whole list and replaying the entrance
   * animation — the visible mid-conversation flash — only for the resume to
   * immediately take over. Hold the replace until the probe settles.
   */
  isFinishProbing: boolean;
  setMessages: (
    updater: UIMessage[] | ((prev: UIMessage[]) => UIMessage[]),
  ) => void;
}

/** Test hook: clear the per-session interrupted-toast ledger. */
export function _resetInterruptedToastLedgerForTests() {
  sessionsWithInterruptedToastShown.clear();
}

/**
 * After a stream ends, replace the in-memory AI SDK messages with the
 * definitive DB state, then keep length-gated top-ups working for later
 * pagination / sync events.
 *
 * **The tricky bit** (hence this being its own hook): `status` flips to
 * "ready" BEFORE the post-turn refetch completes — there's a ~500 ms delay
 * plus the DB round trip. If we force-hydrate immediately, `hydratedMessages`
 * is the STALE pre-turn snapshot and we'd drop any newly-persisted row
 * (mid-turn follow-up user rows, for example).
 *
 * Solution: when the stream ends, snapshot the `hydratedMessages` reference
 * that was current at that moment, and refuse to force-hydrate until React
 * Query swaps in a new reference. Once we see a fresh reference, replace
 * and clear the flag.
 *
 * **Zombie-part recovery**: if the DB's view of the last assistant message
 * has parts still in ``streaming`` / ``input-streaming`` /
 * ``input-available`` AND the backend confirms no active stream, resolve
 * them via ``resolveInterruptedMessage`` and surface a passive toast.
 * This covers the case where the user switched away mid-stream, the
 * backend crashed before finalising, and the user returned to a session
 * the UI would otherwise render with permanent spinners.
 */
export function useHydrateOnStreamEnd({
  sessionId,
  status,
  hydratedMessages,
  isReconnectScheduled,
  hasActiveStream,
  isFinishProbing,
  setMessages,
}: Args) {
  const prevStatusRef = useRef(status);
  const needsForceHydrateRef = useRef(false);
  const staleRefAtStreamEnd = useRef<typeof hydratedMessages | null>(null);
  // A stream that failed since the last replace may have been refused before
  // its send was saved, even when a retry ended cleanly after it.
  const streamFailedSinceReplaceRef = useRef(false);

  // Arm the force-hydrate flag the moment the stream transitions to idle.
  useEffect(() => {
    const wasActive =
      prevStatusRef.current === "streaming" ||
      prevStatusRef.current === "submitted";
    const isNowIdle = status === "ready" || status === "error";
    prevStatusRef.current = status;
    if (wasActive && isNowIdle) {
      needsForceHydrateRef.current = true;
      staleRefAtStreamEnd.current = hydratedMessages ?? null;
      if (status === "error") streamFailedSinceReplaceRef.current = true;
    }
  }, [status, hydratedMessages]);

  // Apply hydration when the right data shows up.
  useEffect(() => {
    if (!hydratedMessages || hydratedMessages.length === 0) return;
    if (status === "streaming" || status === "submitted") return;
    if (isReconnectScheduled) return;

    const deduped = deduplicateMessages(hydratedMessages);
    const needsZombieRecovery =
      !hasActiveStream &&
      hasInProgressAssistantParts(deduped[deduped.length - 1]);
    const finalized = needsZombieRecovery
      ? resolveInterruptedMessage(deduped)
      : deduped;

    // The stale-ref guard below holds back the force-hydrate apply until
    // React Query swaps in a fresh reference.  The interrupted-toast must
    // gate on the same condition: firing earlier would surface "previous
    // response was interrupted" against the pre-turn snapshot — which may
    // not actually have zombie parts once the refetch lands — confusing
    // the user with a toast for state they cannot see yet.
    const isStaleForceHydrateSnapshot =
      needsForceHydrateRef.current &&
      hydratedMessages === staleRefAtStreamEnd.current;

    if (
      needsZombieRecovery &&
      sessionId &&
      !isStaleForceHydrateSnapshot &&
      !sessionsWithInterruptedToastShown.has(sessionId)
    ) {
      sessionsWithInterruptedToastShown.add(sessionId);
      toast({
        title: "Previous response was interrupted",
        description:
          "The chat disconnected before the last response finished. Resend to try again.",
      });
    }

    // Hold ALL message replacement while the post-finish probe decides
    // whether a continuation turn is starting — replacing now would swap
    // every message id and remount the list mid-conversation. This gate is
    // deliberately wider than the `hasActiveStream` one below (which only
    // holds the force-hydrate): it also holds the length-gated top-up. The
    // interrupted toast above is NOT held — an interrupted turn must surface
    // immediately, not after the probe loop finishes.
    if (isFinishProbing) return;

    if (needsForceHydrateRef.current) {
      if (isStaleForceHydrateSnapshot) {
        // Still the pre-turn snapshot — wait for the refetch.
        return;
      }
      // The fresh session data shows the backend still has a live stream
      // (continuation turn dispatching, or a resume about to start). The
      // resume effect in `useCopilotStream` reconnects while the backend
      // reports an active stream, and a merely-stale flag clears on the next
      // session refetch — either way this effect re-fires and the replace
      // lands once the backend goes idle, so no timeout fallback is needed.
      if (hasActiveStream) return;
      const sendsWereTaken = !streamFailedSinceReplaceRef.current;
      setMessages((prev) => {
        const replaced = preservePromotedUserBubbles(
          prev,
          preserveStoppedAssistantMessage(
            prev,
            retainOlderHistory(prev, finalized),
          ),
        );
        return sendsWereTaken
          ? preserveUnpersistedUserBubbles(
              prev,
              replaced,
              firstSequence(finalized),
            )
          : replaced;
      });
      needsForceHydrateRef.current = false;
      staleRefAtStreamEnd.current = null;
      streamFailedSinceReplaceRef.current = false;
      return;
    }

    // Regular length-gated top-up (e.g. pagination brought in older messages).
    setMessages((prev) => (prev.length >= finalized.length ? prev : finalized));
  }, [
    hydratedMessages,
    setMessages,
    status,
    isReconnectScheduled,
    hasActiveStream,
    isFinishProbing,
  ]);
}
