import type { UIMessage } from "ai";
import { useEffect, useRef } from "react";

import { deduplicateMessages } from "./helpers";

type ChatStatus = "submitted" | "streaming" | "ready" | "error";

interface Args {
  status: ChatStatus;
  hydratedMessages: UIMessage[] | undefined;
  isReconnectScheduled: boolean;
  setMessages: (
    updater: UIMessage[] | ((prev: UIMessage[]) => UIMessage[]),
  ) => void;
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
 */
export function useHydrateOnStreamEnd({
  status,
  hydratedMessages,
  isReconnectScheduled,
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

    if (needsForceHydrateRef.current) {
      if (hydratedMessages === staleRefAtStreamEnd.current) {
        // Still the pre-turn snapshot — wait for the refetch.
        return;
      }
      setMessages(deduplicateMessages(hydratedMessages));
      needsForceHydrateRef.current = false;
      staleRefAtStreamEnd.current = null;
      return;
    }

    // Regular length-gated top-up (e.g. pagination brought in older messages).
    setMessages((prev) =>
      prev.length >= hydratedMessages.length
        ? prev
        : deduplicateMessages(hydratedMessages),
    );
  }, [hydratedMessages, setMessages, status, isReconnectScheduled]);
}
