"use client";

import { useEffect, useState } from "react";

/**
 * Returns `Date.now()`, refreshing every `intervalMs` while `enabled`.
 * Drives live elapsed-time UIs for in-flight graph executions without
 * waiting for backend stats updates (see #9690 Part 2).
 */
export function useNow(intervalMs = 1000, enabled = true): number {
  const [now, setNow] = useState(() => Date.now());

  useEffect(() => {
    if (!enabled) return;
    setNow(Date.now());
    const id = setInterval(() => setNow(Date.now()), intervalMs);
    return () => clearInterval(id);
  }, [intervalMs, enabled]);

  return now;
}
