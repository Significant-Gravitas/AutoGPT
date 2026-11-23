"use client";

import { useEffect, useState } from "react";
import { EXIT_MS, MIN_VISIBLE_MS, now } from "./helpers";

export type ThinkingPhase = "hidden" | "shown" | "leaving";

interface Presence {
  phase: ThinkingPhase;
  shownAt: number;
}

/** Keeps the indicator on screen for at least `MIN_VISIBLE_MS` once shown,
 *  then lets it fade for `EXIT_MS` before it unmounts. Wanted again while it
 *  fades, it comes straight back without remounting. `instant` drops both:
 *  under reduced motion, or when another surface takes over the wait. */
export function useThinkingPresence(wanted: boolean, instant: boolean) {
  const [presence, setPresence] = useState<Presence>(() => ({
    phase: wanted ? "shown" : "hidden",
    shownAt: wanted ? now() : 0,
  }));
  let current = presence;
  if (wanted && presence.phase !== "shown") {
    current = {
      phase: "shown",
      shownAt: presence.phase === "hidden" ? now() : presence.shownAt,
    };
  } else if (!wanted && instant && presence.phase !== "hidden") {
    current = { ...presence, phase: "hidden" };
  }
  if (current !== presence) setPresence(current);

  const { phase, shownAt } = current;
  useEffect(
    function leaveOnceShownLongEnough() {
      if (wanted || phase === "hidden") return;
      const next: ThinkingPhase = phase === "shown" ? "leaving" : "hidden";
      const delay =
        phase === "shown"
          ? Math.max(0, MIN_VISIBLE_MS - (now() - shownAt))
          : EXIT_MS;
      const timer = setTimeout(
        () => setPresence((prev) => ({ ...prev, phase: next })),
        delay,
      );
      return () => clearTimeout(timer);
    },
    [wanted, phase, shownAt],
  );

  return phase;
}
