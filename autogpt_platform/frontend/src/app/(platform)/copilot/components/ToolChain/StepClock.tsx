"use client";

import { formatElapsed } from "../JobStatsBar/formatElapsed";
import { useElapsedTimer } from "../JobStatsBar/useElapsedTimer";

const SHOW_TIME_AFTER_SECONDS = 20;

export function StepClock() {
  const { elapsedSeconds } = useElapsedTimer(true);
  if (elapsedSeconds < SHOW_TIME_AFTER_SECONDS) return null;
  return (
    <span className="shrink-0 font-mono text-xs tabular-nums text-foreground-subtle">
      {formatElapsed(elapsedSeconds)}
    </span>
  );
}
