import { useMountEffect } from "@/hooks/useMountEffect";
import { useRef, useState } from "react";
import {
  INITIAL_PROGRESS,
  PHASE_CURVE,
  finishProgress,
  phaseProgress,
  tauForTokens,
  type CompactionPhase,
} from "./helpers";

export function useCompactionProgress(
  phase: CompactionPhase | null,
  tokensBefore: number | undefined,
) {
  const [progress, setProgress] = useState(INITIAL_PROGRESS);
  const [elapsedSeconds, setElapsedSeconds] = useState(0);

  // Read during render so the rAF loop always sees the current phase without
  // a dependency-keyed effect restarting the animation mid-flight.
  const phaseRef = useRef(phase);
  const tokensRef = useRef(tokensBefore);
  const baseRef = useRef(INITIAL_PROGRESS);
  const phaseStartRef = useRef<number | null>(null);
  const lastPhaseRef = useRef<CompactionPhase | null>(phase);
  const progressRef = useRef(INITIAL_PROGRESS);
  phaseRef.current = phase;
  tokensRef.current = tokensBefore;

  useMountEffect(() => {
    let frame = 0;
    const mountedAt = performance.now();

    function tick(now: number) {
      const current = phaseRef.current;

      if (current !== lastPhaseRef.current) {
        lastPhaseRef.current = current;
        baseRef.current = progressRef.current;
        phaseStartRef.current = now;
      }
      if (phaseStartRef.current === null) phaseStartRef.current = now;

      const sincePhase = now - phaseStartRef.current;
      let next = progressRef.current;

      if (current === "done") {
        next = finishProgress(baseRef.current, sincePhase);
      } else if (current !== null) {
        const curve = PHASE_CURVE[current];
        const tau =
          current === "summarizing"
            ? tauForTokens(tokensRef.current)
            : curve.tauMs;
        next = phaseProgress(baseRef.current, curve.cap, sincePhase, tau);
      }

      // The bar is a promise to the user: it never goes backwards.
      next = Math.max(progressRef.current, next);
      progressRef.current = next;
      setProgress(next);
      setElapsedSeconds(Math.floor((now - mountedAt) / 1000));

      if (next < 1) frame = requestAnimationFrame(tick);
    }

    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  });

  return { progress, elapsedSeconds };
}
