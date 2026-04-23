import { useEffect, useRef, useState } from "react";

/**
 * Tick a second-resolution elapsed counter while ``isRunning`` is true.
 *
 * ``startedAtIso`` lets the caller seed the start time from the backend's
 * view of the turn (see ``ActiveStreamInfo.started_at``). Without it the
 * counter would start from 0 on every fresh mount — which is misleading
 * when the user returns to a session that's been running for ``N`` seconds
 * on the backend and expects to see ``N`` seconds, not zero.
 *
 * The baseline is captured on the ``false`` → ``true`` transition of
 * ``isRunning``. Later changes to ``startedAtIso`` are ignored until the
 * timer resets (``isRunning`` goes back to ``false``), so we don't
 * re-anchor mid-turn.
 */
export function useElapsedTimer(
  isRunning: boolean,
  startedAtIso?: string | null,
) {
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const startTimeRef = useRef<number | null>(null);
  const intervalRef = useRef<ReturnType<typeof setInterval>>();
  const startedAtIsoRef = useRef(startedAtIso);
  startedAtIsoRef.current = startedAtIso;

  useEffect(() => {
    if (isRunning) {
      if (startTimeRef.current === null) {
        const seed = resolveStartMs(startedAtIsoRef.current);
        startTimeRef.current = seed;
        setElapsedSeconds(Math.max(0, Math.floor((Date.now() - seed) / 1000)));
      }

      intervalRef.current = setInterval(() => {
        if (startTimeRef.current !== null) {
          setElapsedSeconds(
            Math.max(0, Math.floor((Date.now() - startTimeRef.current) / 1000)),
          );
        }
      }, 1000);

      return () => clearInterval(intervalRef.current);
    }

    clearInterval(intervalRef.current);
    startTimeRef.current = null;
  }, [isRunning]);

  return { elapsedSeconds };
}

function resolveStartMs(iso: string | null | undefined): number {
  if (!iso) return Date.now();
  const parsed = new Date(iso).getTime();
  return Number.isFinite(parsed) ? parsed : Date.now();
}
