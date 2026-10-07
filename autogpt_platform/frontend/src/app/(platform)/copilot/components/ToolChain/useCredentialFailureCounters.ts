"use client";

import { useEffect, useRef } from "react";
import { trackCredentialConnectionFailure } from "@/services/credentials/connection-analytics";
import type { ChainActionEntry } from "./chainActions";

// A just-connected card receives its credential through an async store reload,
// so a brief unwired window is the happy path. Past this it is stuck.
const STUCK_AFTER_CONNECT_MS = 5000;

interface Args {
  entries: ReadonlyMap<string, ChainActionEntry>;
}

/** Counts sign-ins that complete and never reach the card that asked for them.
 *  Keyed on the credential alone — a card is legitimately not `ready` while the
 *  user is still filling in its run inputs. */
export function useCredentialFailureCounters({ entries }: Args) {
  const stuckTimers = useRef(new Map<string, ReturnType<typeof setTimeout>>());
  const stuckReported = useRef(new Set<string>());

  useEffect(
    function reportCardsStuckAfterConnecting() {
      const timers = stuckTimers.current;
      const reported = stuckReported.current;
      // An unregistered card left the chain; its pending timer would count a
      // user who is no longer looking at it.
      timers.forEach((timer, id) => {
        if (entries.has(id)) return;
        clearTimeout(timer);
        timers.delete(id);
      });
      entries.forEach((entry) => {
        if (entry.justConnected && entry.credentialsReady === false) {
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
}
