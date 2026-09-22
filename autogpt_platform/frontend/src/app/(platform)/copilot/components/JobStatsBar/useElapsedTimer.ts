import { useEffect, useRef, useState } from "react";

/**
 * Ticks once per second while `isRunning` is true.
 *
 * Pass `anchorIso` (a server-issued ISO timestamp, e.g. the active stream's
 * `started_at` or the last user/tool message's `createdAt`) to count from
 * that absolute wall-clock point instead of from when this hook first saw
 * `isRunning = true`. This is what makes the "Considering Xs" counter
 * survive a page refresh mid-turn — it reflects actual elapsed time since
 * the turn's last recorded activity, not the moment the current browser
 * tab mounted.
 */
export function useElapsedTimer(isRunning: boolean, anchorIso?: string | null) {
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const startTimeRef = useRef<number | null>(null);
  const intervalRef = useRef<ReturnType<typeof setInterval>>();

  useEffect(() => {
    if (isRunning) {
      // Re-sync on every re-run so a late-arriving anchorIso (e.g. session
      // data loads after the timer started on page refresh) updates the
      // start time instead of being ignored.
      const anchorMs = anchorIso ? Date.parse(anchorIso) : NaN;
      startTimeRef.current = Number.isFinite(anchorMs) ? anchorMs : Date.now();
      setElapsedSeconds(
        Math.max(0, Math.floor((Date.now() - startTimeRef.current) / 1000)),
      );

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
  }, [isRunning, anchorIso]);

  return { elapsedSeconds };
}
