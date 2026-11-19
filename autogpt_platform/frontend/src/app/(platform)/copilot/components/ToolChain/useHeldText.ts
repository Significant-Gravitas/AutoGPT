"use client";

import { useEffect, useState } from "react";

interface Shown {
  text: string;
  at: number;
}

function now() {
  return Date.now();
}

/** The text to show for a label that may change faster than it can be read.
 *  Each text stays up for at least `minHoldMs`; a change inside that window
 *  waits for it to end, and only the newest change is shown then, so a burst
 *  of fast tools steps through one label instead of flickering through all
 *  of them. A `minHoldMs` of 0 shows every change at once. */
export function useHeldText(text: string, minHoldMs: number) {
  const [shown, setShown] = useState<Shown>(() => ({ text, at: now() }));
  let current = shown;
  if (
    shown.text !== text &&
    (minHoldMs <= 0 || now() - shown.at >= minHoldMs)
  ) {
    current = { text, at: now() };
    setShown(current);
  }
  const pending = current.text !== text;

  useEffect(
    function showLatestOnceHeld() {
      if (!pending) return;
      const timer = setTimeout(
        () => setShown({ text, at: now() }),
        Math.max(0, minHoldMs - (now() - shown.at)),
      );
      return () => clearTimeout(timer);
    },
    [pending, text, minHoldMs, shown.at],
  );

  return current.text;
}
