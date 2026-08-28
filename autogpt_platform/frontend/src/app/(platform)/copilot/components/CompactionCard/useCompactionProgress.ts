import { useMountEffect } from "@/hooks/useMountEffect";
import { useRef, useState } from "react";
import {
  INITIAL_PROGRESS,
  PARKED_POLL_MS,
  PHASE_CURVE,
  SETTLE_EPSILON,
  phaseProgress,
  tauForTokens,
  type CompactionPhase,
} from "./helpers";

export function useCompactionProgress(
  phase: CompactionPhase,
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
  const lastPhaseRef = useRef<CompactionPhase>(phase);
  const progressRef = useRef(INITIAL_PROGRESS);
  const percentRef = useRef(Math.round(INITIAL_PROGRESS * 100));
  const secondsRef = useRef(0);
  phaseRef.current = phase;
  tokensRef.current = tokensBefore;

  useMountEffect(() => {
    let frame = 0;
    let parkedTimer: ReturnType<typeof setTimeout> | undefined;
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
      const curve = PHASE_CURVE[current];
      const tau =
        current === "summarizing"
          ? tauForTokens(tokensRef.current)
          : curve.tauMs;
      let next = phaseProgress(baseRef.current, curve.cap, sincePhase, tau);

      // The bar is a promise to the user: it never goes backwards.
      next = Math.max(progressRef.current, next);
      progressRef.current = next;

      // Quantize commits: only a change the DOM can show (a whole percent, a
      // whole second) triggers a render, not 60 identical frames a second.
      const percent = Math.round(next * 100);
      if (percent !== percentRef.current) {
        percentRef.current = percent;
        setProgress(next);
      }
      const seconds = Math.floor((now - mountedAt) / 1000);
      if (seconds !== secondsRef.current) {
        secondsRef.current = seconds;
        setElapsedSeconds(seconds);
      }

      // Once the curve is visually pinned at its ceiling there is nothing
      // left to animate, so drop off the frame clock and idle on a slow
      // timer instead — a stalled rebuild must not hold the tab at 60Hz for
      // minutes. The timer keeps the elapsed seconds honest and picks the
      // frame loop back up when a later phase raises the ceiling, so the bar
      // still never finishes before the work does.
      if (curve.cap - next <= SETTLE_EPSILON) {
        parkedTimer = setTimeout(() => {
          frame = requestAnimationFrame(tick);
        }, PARKED_POLL_MS);
        return;
      }
      frame = requestAnimationFrame(tick);
    }

    frame = requestAnimationFrame(tick);
    return () => {
      cancelAnimationFrame(frame);
      if (parkedTimer !== undefined) clearTimeout(parkedTimer);
    };
  });

  return { progress, elapsedSeconds };
}
