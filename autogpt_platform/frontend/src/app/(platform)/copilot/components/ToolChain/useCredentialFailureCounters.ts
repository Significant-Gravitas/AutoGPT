"use client";

import { useEffect, useRef } from "react";
import { trackCredentialConnectionFailure } from "@/services/credentials/connection-analytics";
import type { ChainActionEntry } from "./chainActions";

// A just-connected card reaches `ready` through an async store reload, so a
// brief not-ready window is the happy path. Past this it is stuck.
const STUCK_AFTER_CONNECT_MS = 5000;

interface Args {
  entries: ReadonlyMap<string, ChainActionEntry>;
  isStreaming: boolean;
  offerProceed: boolean;
  justConnectedHere: boolean;
}

/** Counts the two ways a chain's Proceed strands the user without throwing:
 *  a card whose sign-in completed and never became ready, and a Proceed
 *  restored from history that drafts "I've configured..." about nothing. */
export function useCredentialFailureCounters({
  entries,
  isStreaming,
  offerProceed,
  justConnectedHere,
}: Args) {
  const stuckTimers = useRef(new Map<string, ReturnType<typeof setTimeout>>());
  const stuckReported = useRef(new Set<string>());
  const everStreamed = useRef(false);
  const everConnected = useRef(false);
  const staleReported = useRef(false);

  useEffect(
    function reportCardsStuckAfterConnecting() {
      const timers = stuckTimers.current;
      const reported = stuckReported.current;
      entries.forEach((entry) => {
        if (entry.justConnected && !entry.ready) {
          if (timers.has(entry.id) || reported.has(entry.id)) return;
          timers.set(
            entry.id,
            setTimeout(() => {
              timers.delete(entry.id);
              reported.add(entry.id);
              trackCredentialConnectionFailure(
                "credential_proceed_stuck_after_connect",
              );
            }, STUCK_AFTER_CONNECT_MS),
          );
          return;
        }
        clearTimeout(timers.get(entry.id));
        timers.delete(entry.id);
      });
    },
    [entries],
  );

  useEffect(function clearStuckTimersOnUnmount() {
    const timers = stuckTimers.current;
    return () => {
      timers.forEach(clearTimeout);
      timers.clear();
    };
  }, []);

  useEffect(
    function reportStaleProceedFromHistory() {
      if (isStreaming) everStreamed.current = true;
      if (justConnectedHere) everConnected.current = true;
      if (staleReported.current || !offerProceed) return;
      // Streamed or connected in this page life: the Proceed is this chain's
      // own, however little it has to say.
      if (everStreamed.current || everConnected.current) return;
      staleReported.current = true;
      trackCredentialConnectionFailure("credential_proceed_stale_from_history");
    },
    [offerProceed, isStreaming, justConnectedHere],
  );
}
