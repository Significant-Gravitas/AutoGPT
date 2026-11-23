"use client";

import { useReducedMotion } from "framer-motion";
import { useEffect, useState } from "react";
import type { ChainRowState } from "./helpers";

/** How long a tool row shows itself running, at the least. A call that takes
 *  80 ms would otherwise flip from spinner to done within a frame or two, or
 *  mount already finished, and read as a row that popped in. */
export const MIN_RUNNING_MS = 400;

interface Args {
  state: ChainRowState;
  /** The row arrived during a live stream already finished, so it never got
   *  to show itself running. Read once, at mount. */
  arrivedFinished: boolean;
  /** Rows this does not apply to (reasoning, narration, held calls, rows
   *  asking for the user) show their state as it is. */
  enabled: boolean;
}

function now() {
  return Date.now();
}

function elapsedSince(startedAt: number) {
  return now() - startedAt;
}

/** The state a tool row shows: its own, except that a call which finished
 *  within `MIN_RUNNING_MS` of starting keeps its running look until that time
 *  is up. Only "done" is held: an error shows the moment it lands. Under
 *  reduced motion every state shows at once. */
export function useMinimumRunning({ state, arrivedFinished, enabled }: Args) {
  const reducedMotion = useReducedMotion();
  const active = enabled && !reducedMotion;
  const [runningSince, setRunningSince] = useState<number | null>(() =>
    state === "running" || arrivedFinished ? now() : null,
  );
  const [holding, setHolding] = useState(
    () => arrivedFinished && state === "done",
  );
  const [prevState, setPrevState] = useState(state);
  if (prevState !== state) {
    setPrevState(state);
    if (state === "running") {
      setRunningSince(now());
      setHolding(false);
    } else if (state === "error") {
      setHolding(false);
    } else if (
      runningSince !== null &&
      elapsedSince(runningSince) < MIN_RUNNING_MS
    ) {
      setHolding(true);
    }
  }

  useEffect(
    function releaseAfterMinimum() {
      if (!holding || runningSince === null) return;
      const timer = setTimeout(
        () => setHolding(false),
        Math.max(0, MIN_RUNNING_MS - elapsedSince(runningSince)),
      );
      return () => clearTimeout(timer);
    },
    [holding, runningSince],
  );

  return active && holding && state === "done" ? "running" : state;
}
