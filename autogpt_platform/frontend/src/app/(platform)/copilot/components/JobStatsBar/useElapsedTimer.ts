import { useNowSecond } from "@/hooks/useNowSecond";
import { useState } from "react";

interface Run {
  isRunning: boolean;
  anchorIso: string | null | undefined;
  startedAtMs: number | null;
  frozenSeconds: number;
}

function resolveStartMs(
  anchorIso: string | null | undefined,
  fallbackMs: number,
) {
  const anchorMs = anchorIso ? Date.parse(anchorIso) : NaN;
  return Number.isFinite(anchorMs) ? anchorMs : fallbackMs;
}

export function getElapsedSeconds(startedAtMs: number, nowMs: number) {
  return Math.max(0, Math.floor((nowMs - startedAtMs) / 1000));
}

/**
 * When the current run started, without ticking. `startedAtMs` is set while
 * `isRunning` is true and null otherwise; `frozenSeconds` holds the elapsed
 * reading at the moment the last run stopped (0 before any run).
 *
 * Pass `anchorIso` (a server-issued ISO timestamp, e.g. the active stream's
 * `started_at` or the last user/tool message's `createdAt`) to count from
 * that absolute wall-clock point instead of from when this hook first saw
 * `isRunning = true`. This is what makes the "Considering Xs" counter
 * survive a page refresh mid-turn — it reflects actual elapsed time since
 * the turn's last recorded activity, not the moment the current browser
 * tab mounted.
 */
export function useElapsedAnchor(
  isRunning: boolean,
  anchorIso?: string | null,
) {
  const [run, setRun] = useState<Run>(() => ({
    isRunning,
    anchorIso,
    startedAtMs: isRunning ? resolveStartMs(anchorIso, Date.now()) : null,
    frozenSeconds: 0,
  }));

  // Re-sync whenever the run starts or stops, and whenever a late-arriving
  // anchorIso (e.g. session data loads after the timer started on page
  // refresh) replaces the one the run started from. An anchor that drops
  // out mid-run (a session refetch clears it) keeps the start it had.
  let current = run;
  if (
    run.isRunning !== isRunning ||
    (isRunning && run.anchorIso !== anchorIso)
  ) {
    current = {
      isRunning,
      anchorIso,
      startedAtMs: isRunning
        ? resolveStartMs(
            anchorIso,
            run.isRunning && run.startedAtMs !== null
              ? run.startedAtMs
              : Date.now(),
          )
        : null,
      frozenSeconds:
        !isRunning && run.startedAtMs !== null
          ? getElapsedSeconds(run.startedAtMs, Date.now())
          : run.frozenSeconds,
    };
    setRun(current);
  }

  return {
    startedAtMs: current.startedAtMs,
    frozenSeconds: current.frozenSeconds,
  };
}

/**
 * Elapsed whole seconds of the current run, ticking once per second on the
 * shared wall-clock second while `isRunning` is true and holding the last
 * reading once it stops. See `useElapsedAnchor` for `anchorIso`.
 */
export function useElapsedTimer(isRunning: boolean, anchorIso?: string | null) {
  const { startedAtMs, frozenSeconds } = useElapsedAnchor(isRunning, anchorIso);
  const nowMs = useNowSecond(isRunning);
  const elapsedSeconds =
    startedAtMs === null
      ? frozenSeconds
      : getElapsedSeconds(startedAtMs, nowMs);
  return { elapsedSeconds };
}
