"use client";

import { useEffect, useState } from "react";

const TICK_MS = 1000;

/** The time now, kept current while `isTicking`, so a countdown on the card
 *  moves with the clock instead of waiting for something else to redraw it. */
export function useNow(isTicking: boolean) {
  const [now, setNow] = useState(Date.now);

  useEffect(() => {
    if (!isTicking) return;
    setNow(Date.now());
    const timer = setInterval(() => setNow(Date.now()), TICK_MS);
    return () => clearInterval(timer);
  }, [isTicking]);

  return new Date(now);
}
