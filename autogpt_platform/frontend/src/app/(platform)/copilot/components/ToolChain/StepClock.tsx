"use client";

import { useEffect, useState } from "react";
import { SHOW_TIME_AFTER_SECONDS } from "../JobStatsBar/constants";
import { formatElapsed } from "../JobStatsBar/formatElapsed";
import { useElapsedTimer } from "../JobStatsBar/useElapsedTimer";
import type { ChainRow } from "./helpers";

const MAX_REMEMBERED_STEPS = 200;
const stepStarts = new Map<string, string>();

// When this tab first saw each running step, kept outside the row so a
// remount (switching chats, expanding the chain) resumes the same count.
// A step seen again moves to the back, so eviction drops the stalest one.
function stepStartedAt(stepKey: string) {
  const known = stepStarts.get(stepKey);
  if (known) {
    stepStarts.delete(stepKey);
    stepStarts.set(stepKey, known);
    return known;
  }
  const startedAt = new Date().toISOString();
  stepStarts.set(stepKey, startedAt);
  if (stepStarts.size > MAX_REMEMBERED_STEPS) {
    const oldest = stepStarts.keys().next().value;
    if (oldest !== undefined) stepStarts.delete(oldest);
  }
  return startedAt;
}

// Every running step's start is noted as soon as the chain has its row, so a
// step outside the streaming window still counts from then once expanded. A
// finished step's start is dropped, shown or not: some providers reuse call
// ids across chats, and a stale start would make a new step look old.
export function useNoteRunningSteps(rows: readonly ChainRow[]) {
  useEffect(() => {
    for (const row of rows) {
      if (row.state === "running") stepStartedAt(row.key);
      else stepStarts.delete(row.key);
    }
  }, [rows]);
}

interface Props {
  stepKey: string;
}

export function StepClock({ stepKey }: Props) {
  const [anchorIso] = useState(() => stepStartedAt(stepKey));
  const { elapsedSeconds } = useElapsedTimer(true, anchorIso);
  if (elapsedSeconds < SHOW_TIME_AFTER_SECONDS) return null;
  return (
    <span
      aria-hidden
      className="shrink-0 font-mono text-xs text-foreground-subtle tabular-nums"
    >
      {formatElapsed(elapsedSeconds)}
    </span>
  );
}
