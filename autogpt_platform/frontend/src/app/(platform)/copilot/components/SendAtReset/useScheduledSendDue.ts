"use client";

import { useEffect, useState } from "react";

/** Whether a scheduled send's time has come: it may be going out or gone,
 *  so it can't be taken back to edit. Flips on its own at that time. */
export function useScheduledSendDue(runAt: Date | string) {
  const [now, setNow] = useState(() => Date.now());
  const due = new Date(runAt).getTime();

  useEffect(
    function flipWhenDue() {
      if (due <= now) return;
      const timer = setTimeout(() => setNow(Date.now()), due - now);
      return () => clearTimeout(timer);
    },
    [due, now],
  );

  return due <= now;
}
