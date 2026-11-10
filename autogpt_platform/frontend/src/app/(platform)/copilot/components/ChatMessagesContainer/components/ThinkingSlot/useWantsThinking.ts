"use client";

import { useEffect, useState } from "react";
import type { TurnActivity } from "../../thinkingActivity";
import { getWait } from "./helpers";

interface Seen {
  key: string | null;
  ready: boolean;
}

/** Whether the turn has gone quiet long enough to show the indicator: at
 *  once when a request is out with nothing back yet, after `STEP_GAP_MS`
 *  between steps, and after `TEXT_IDLE_MS` once text stops growing. */
export function useWantsThinking(activity: TurnActivity, enabled: boolean) {
  const { key, delayMs, keepsShowing } = getWait(activity, enabled);
  const [seen, setSeen] = useState<Seen>(() => ({
    key,
    ready: key !== null && delayMs <= 0,
  }));
  let current = seen;
  if (seen.key !== key) {
    current = {
      key,
      ready: key !== null && (delayMs <= 0 || (keepsShowing && seen.ready)),
    };
    setSeen(current);
  }

  const waitingKey = current.ready ? null : current.key;
  useEffect(
    function showOnceTheWaitHolds() {
      if (waitingKey === null) return;
      const timer = setTimeout(
        () =>
          setSeen((prev) =>
            prev.key === waitingKey ? { ...prev, ready: true } : prev,
          ),
        delayMs,
      );
      return () => clearTimeout(timer);
    },
    [waitingKey, delayMs],
  );

  return current.ready;
}
