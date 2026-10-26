"use client";

import { useEffect, useState } from "react";

const TICK_MS = 1000;

/** The time now, kept current until `until` (epoch ms) passes, so a countdown
 *  on the card moves with the clock and stops once there is nothing left to
 *  count. `null` never ticks. */
export function useNow(until: number | null) {
  const [now, setNow] = useState(Date.now);
  const isTicking = until !== null && now < until;

  useEffect(() => {
    if (!isTicking) return;
    setNow(Date.now());
    const timer = setInterval(() => setNow(Date.now()), TICK_MS);
    return () => clearInterval(timer);
  }, [isTicking]);

  return new Date(now);
}
