import { useEffect, useRef, useState } from "react";

/**
 * Formats elapsed seconds into a human-readable duration string.
 * Examples: "0s", "45s", "2m 15s", "1h 5m 20s"
 */
function formatElapsed(totalSeconds: number): string {
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = totalSeconds % 60;

  if (hours > 0) {
    return `${hours}h ${minutes}m ${seconds}s`;
  }
  if (minutes > 0) {
    return `${minutes}m ${seconds}s`;
  }
  return `${seconds}s`;
}

interface UseJobTimerArgs {
  /** True when the chat is actively streaming or submitted. */
  isActive: boolean;
}

export function useJobTimer({ isActive }: UseJobTimerArgs) {
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const startTimeRef = useRef<number | null>(null);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(
    function manageTimer() {
      if (isActive) {
        // Start the timer
        startTimeRef.current = Date.now();
        setElapsedSeconds(0);

        intervalRef.current = setInterval(function tick() {
          if (startTimeRef.current !== null) {
            const elapsed = Math.floor(
              (Date.now() - startTimeRef.current) / 1000,
            );
            setElapsedSeconds(elapsed);
          }
        }, 1000);
      } else {
        // Stop the timer, freeze the displayed value
        if (intervalRef.current !== null) {
          clearInterval(intervalRef.current);
          intervalRef.current = null;
        }
      }

      return function cleanup() {
        if (intervalRef.current !== null) {
          clearInterval(intervalRef.current);
          intervalRef.current = null;
        }
      };
    },
    [isActive],
  );

  const formattedTime = formatElapsed(elapsedSeconds);
  const hasStarted = startTimeRef.current !== null;

  return {
    elapsedSeconds,
    formattedTime,
    isRunning: isActive,
    hasStarted,
  };
}
