import { toast } from "@/components/molecules/Toast/use-toast";
import type { UIMessage } from "ai";
import { useEffect, useRef } from "react";

import {
  deduplicateMessages,
  hasInProgressAssistantParts,
  resolveInterruptedMessage,
} from "./helpers";
import { extractDbSequence } from "./helpers/convertChatSessionToUiMessages";

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
  let oldestHydratedSeq: number | null = null;
  for (const message of hydrated) {
    oldestHydratedSeq = extractDbSequence(message);
    if (oldestHydratedSeq !== null) break;
  }
  if (oldestHydratedSeq === null) return hydrated;
  const windowStart = oldestHydratedSeq;

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
  const promoted = prev.filter(
    (m) =>
      typeof m.id === "string" && m.id.startsWith(PROMOTED_BUBBLE_ID_PREFIX),
  );
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
  setMessages,
}: Args) {
  const prevStatusRef = useRef(status);
  const needsForceHydrateRef = useRef(false);
  const staleRefAtStreamEnd = useRef<typeof hydratedMessages | null>(null);

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

    if (needsForceHydrateRef.current) {
      if (isStaleForceHydrateSnapshot) {
        // Still the pre-turn snapshot — wait for the refetch.
        return;
      }
      setMessages((prev) =>
        preservePromotedUserBubbles(prev, retainOlderHistory(prev, finalized)),
      );
      needsForceHydrateRef.current = false;
      staleRefAtStreamEnd.current = null;
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
  ]);
}
