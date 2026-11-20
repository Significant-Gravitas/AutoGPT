import { type MotionValue, motionValue } from "framer-motion";
import { useEffect, useRef } from "react";

// A wandering loudness with pauses, close enough to speech for a preview.
export function useSimulatedVoice(isActive: boolean) {
  const levelsRef = useRef<MotionValue<number>[] | null>(null);
  if (levelsRef.current === null) {
    levelsRef.current = Array.from({ length: 5 }, () => motionValue(0));
  }
  const levels = levelsRef.current;

  useEffect(() => {
    if (!isActive) {
      levels.forEach((level) => level.set(0));
      return;
    }
    const start = performance.now();
    const timer = setInterval(() => {
      const t = (performance.now() - start) / 1000;
      const phrase = Math.max(0, Math.sin(t * 0.9)) * 0.9;
      levels.forEach((level, index) => {
        const wobble = 0.5 + 0.5 * Math.sin(t * (5 + index * 1.7) + index);
        level.set(phrase * (0.35 + 0.65 * wobble));
      });
    }, 60);
    return () => clearInterval(timer);
  }, [isActive, levels]);

  return levels;
}
