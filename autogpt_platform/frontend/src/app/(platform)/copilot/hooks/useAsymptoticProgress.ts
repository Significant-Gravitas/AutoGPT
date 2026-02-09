import { useEffect, useRef, useState } from "react";

/**
 * Hook that returns a progress value that starts fast and slows down,
 * asymptotically approaching but never reaching the max value.
 *
 * Uses a half-life formula: progress = max * (1 - 0.5^(time/halfLife))
 * This creates a "loading bar" effect where:
 * - 50% is reached at halfLifeSeconds
 * - 75% is reached at 2 * halfLifeSeconds
 * - 87.5% is reached at 3 * halfLifeSeconds
 *
 * @param isActive - Whether the progress should be animating
 * @param halfLifeSeconds - Time in seconds to reach 50% progress (default: 30)
 * @param maxProgress - Maximum progress value to approach (default: 100)
 * @param intervalMs - Update interval in milliseconds (default: 100)
 * @returns Current progress value (0–maxProgress)
 */
export function useAsymptoticProgress(
  isActive: boolean,
  halfLifeSeconds = 30,
  maxProgress = 100,
  intervalMs = 100,
) {
  const [progress, setProgress] = useState(0);
  const elapsedTimeRef = useRef(0);

  useEffect(() => {
    if (!isActive) {
      setProgress(0);
      elapsedTimeRef.current = 0;
      return;
    }

    const interval = setInterval(() => {
      elapsedTimeRef.current += intervalMs / 1000;
      const newProgress =
        maxProgress *
        (1 - Math.pow(0.5, elapsedTimeRef.current / halfLifeSeconds));
      setProgress(newProgress);
    }, intervalMs);

    return () => clearInterval(interval);
  }, [isActive, halfLifeSeconds, maxProgress, intervalMs]);

  return progress;
}
