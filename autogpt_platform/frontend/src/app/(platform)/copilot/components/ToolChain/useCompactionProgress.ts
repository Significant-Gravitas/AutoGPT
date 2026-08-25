import { useEffect, useRef, useState } from "react";

/** Compaction duration is unknown (LLM summarization, often 20-90s), so the
 *  bar saturates toward a cap it never reaches: every moment it covers a
 *  fixed fraction of the REMAINING distance. Fast early movement, slows
 *  forever, can't finish before the work does. */
const CAP = 0.92;
const TAU_MS = 15_000;

/** When the real completion lands, sprint the remaining distance with the
 *  same exponential shape, just much faster. */
const FINISH_TAU_MS = 120;
const FINISH_SNAP = 0.995;

export function asymptoticProgress(elapsedMs: number): number {
  return CAP * (1 - Math.exp(-Math.max(0, elapsedMs) / TAU_MS));
}

export function finishProgress(from: number, sinceDoneMs: number): number {
  const next = 1 - (1 - from) * Math.exp(-Math.max(0, sinceDoneMs) / FINISH_TAU_MS);
  return next >= FINISH_SNAP ? 1 : next;
}

/** 0..1 progress for a compaction row. While running it follows the
 *  asymptotic curve; when `done` flips it interpolates to 100% from
 *  wherever it was. Mounted already-done (history replay) → 1 instantly. */
export function useCompactionProgress(done: boolean): number {
  const [progress, setProgress] = useState(done ? 1 : 0);
  const progressRef = useRef(progress);
  const startRef = useRef<number | null>(null);
  const finishRef = useRef<{ at: number; from: number } | null>(null);

  useEffect(() => {
    if (progressRef.current >= 1) return;
    let raf = 0;
    const tick = (now: number) => {
      let next: number;
      if (done) {
        finishRef.current ??= { at: now, from: progressRef.current };
        next = finishProgress(finishRef.current.from, now - finishRef.current.at);
      } else {
        startRef.current ??= now;
        next = asymptoticProgress(now - startRef.current);
      }
      progressRef.current = next;
      setProgress(next);
      if (next < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [done]);

  return progress;
}
