import { useEffect, useRef, useState } from "react";

/**
 * Cubic Ease Out easing function: 1 - (1 - t)^3
 * Starts fast and decelerates smoothly to a stop.
 */
function cubicEaseOut(t: number): number {
  return 1 - Math.pow(1 - t, 3);
}

export interface AsymptoticProgressResult {
  progress: number;
  isAnimationDone: boolean;
}

/**
 * Hook that returns a progress value that starts fast and slows down,
 * asymptotically approaching but never reaching the max value.
 *
 * Uses a half-life formula: progress = max * (1 - 0.5^(time/halfLife))
 * This creates the "game loading bar" effect where:
 * - 50% is reached at halfLifeSeconds
 * - 75% is reached at 2 * halfLifeSeconds
 * - 87.5% is reached at 3 * halfLifeSeconds
 * - and so on...
 *
 * When isComplete is set to true, animates from current progress to 100%
 * using Cubic Ease Out over 300ms.
 *
 * @param isActive - Whether the progress should be animating
 * @param isComplete - Whether to animate to 100% (completion animation)
 * @param halfLifeSeconds - Time in seconds to reach 50% progress (default: 30)
 * @param maxProgress - Maximum progress value to approach (default: 100)
 * @param intervalMs - Update interval in milliseconds (default: 100)
 * @returns Object with current progress value and whether completion animation is done
 */
export function useAsymptoticProgress(
  isActive: boolean,
  isComplete = false,
  halfLifeSeconds = 30,
  maxProgress = 100,
  intervalMs = 100,
): AsymptoticProgressResult {
  const [progress, setProgress] = useState(0);
  const [isAnimationDone, setIsAnimationDone] = useState(false);
  const elapsedTimeRef = useRef(0);
  const completionStartProgressRef = useRef<number | null>(null);
  const animationFrameRef = useRef<number | null>(null);

  // Handle asymptotic progress when active but not complete
  useEffect(() => {
    if (!isActive || isComplete) {
      if (!isComplete) {
        setProgress(0);
        elapsedTimeRef.current = 0;
        setIsAnimationDone(false);
        completionStartProgressRef.current = null;
      }
      return;
    }

    const interval = setInterval(() => {
      elapsedTimeRef.current += intervalMs / 1000;
      // Half-life approach: progress = max * (1 - 0.5^(time/halfLife))
      // At t=halfLife: 50%, at t=2*halfLife: 75%, at t=3*halfLife: 87.5%, etc.
      const newProgress =
        maxProgress *
        (1 - Math.pow(0.5, elapsedTimeRef.current / halfLifeSeconds));
      setProgress(newProgress);
    }, intervalMs);

    return () => clearInterval(interval);
  }, [isActive, isComplete, halfLifeSeconds, maxProgress, intervalMs]);

  // Handle completion animation
  useEffect(() => {
    if (!isComplete) {
      return;
    }

    // Capture the starting progress when completion begins
    if (completionStartProgressRef.current === null) {
      completionStartProgressRef.current = progress;
    }

    const startProgress = completionStartProgressRef.current;
    const animationDuration = 300; // 300ms
    const startTime = performance.now();

    function animate(currentTime: number) {
      const elapsed = currentTime - startTime;
      const t = Math.min(elapsed / animationDuration, 1);

      // Cubic Ease Out from current progress to maxProgress
      const easedProgress =
        startProgress + (maxProgress - startProgress) * cubicEaseOut(t);
      setProgress(easedProgress);

      if (t < 1) {
        animationFrameRef.current = requestAnimationFrame(animate);
      } else {
        setProgress(maxProgress);
        setIsAnimationDone(true);
      }
    }

    animationFrameRef.current = requestAnimationFrame(animate);

    return () => {
      if (animationFrameRef.current !== null) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [isComplete, maxProgress]);

  return { progress, isAnimationDone };
}
