"use client";

import { useEffect } from "react";
import {
  getConsent,
  subscribeToConsent,
  type ConsentCategory,
  type ConsentState,
} from "./consent";

// A script that already ran (DataFast, Vercel Analytics, session replay)
// cannot be unloaded, so taking a category back reloads the page to shed it.
// Granting needs no reload: consumers pick it up through useConsent.
export function ConsentWithdrawalReload() {
  useEffect(() => {
    let previous = getConsent();
    return subscribeToConsent(() => {
      const next = getConsent();
      const withdrawn = wasWithdrawn(previous, next);
      previous = next;
      if (withdrawn) window.location.reload();
    });
  }, []);

  return null;
}

function wasWithdrawn(previous: ConsentState, next: ConsentState): boolean {
  return (Object.keys(previous) as ConsentCategory[]).some(
    (category) => previous[category] && !next[category],
  );
}
