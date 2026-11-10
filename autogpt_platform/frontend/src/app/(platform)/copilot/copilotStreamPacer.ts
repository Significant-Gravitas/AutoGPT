/**
 * Time-based pacing for streamed text: how much of the buffered backlog is
 * due at a given moment.
 *
 * Each arrival extends a deadline by the time its characters take at the
 * steady rate, capped so nothing waits more than `MAX_LAG_MS` after it
 * arrived. Between arrivals the backlog drains in a straight line towards
 * that deadline. So a slow model's text flows at the steady rate, a fast
 * model's runs at most `MAX_LAG_MS` behind, and a whole block sent in one
 * delta flows over `MAX_LAG_MS` instead of appearing at once. Pacing reads
 * the clock, not the tick count, so a late timer catches up in one step.
 */

/** About 70 words a second: fast to read along with, still visibly flowing. */
export const STEADY_CHARS_PER_SECOND = 400;
/** The furthest the shown text falls behind what has arrived. */
export const MAX_LAG_MS = 800;
/** One frame between reveals. */
export const TICK_MS = 16;
/** With reduced motion, fewer and larger steps at the same rate. */
export const REDUCED_MOTION_TICK_MS = 120;

export interface TextPacer {
  /** `chars` characters arrived at `now`. */
  arrive(chars: number, now: number): void;
  /** A drain starts at `now`: time spent idle before it is not credit. */
  begin(now: number): void;
  /** The furthest of `cuts` (ascending offsets) due at `now`, or 0. */
  take(cuts: readonly number[], now: number): number;
}

export function createTextPacer(): TextPacer {
  let deadline = 0;
  let lastTick = 0;
  // Characters already due but not shown, because no cut fell inside them.
  let credit = 0;

  return {
    arrive(chars, now) {
      const queued =
        Math.max(deadline, now) + (chars * 1000) / STEADY_CHARS_PER_SECOND;
      deadline = Math.min(queued, now + MAX_LAG_MS);
    },
    begin(now) {
      lastTick = now;
      credit = 0;
    },
    take(cuts, now) {
      const backlog = cuts[cuts.length - 1] ?? 0;
      const elapsed = Math.max(0, now - lastTick);
      lastTick = now;
      const remaining = deadline - now;
      const share = remaining <= 0 ? 1 : elapsed / (elapsed + remaining);
      const allowed = credit + backlog * share;
      let cut = 0;
      for (const candidate of cuts) {
        if (candidate > allowed + 1e-9) break;
        cut = candidate;
      }
      credit = cut === backlog ? 0 : allowed - cut;
      return cut;
    },
  };
}

// The browser defaults the smoothing transform runs on.
export function defaultWait(ms: number) {
  return new Promise<void>((resolve) => setTimeout(resolve, ms));
}

export function defaultNow() {
  return typeof performance !== "undefined" ? performance.now() : Date.now();
}

export function isDocumentHidden() {
  return (
    typeof document !== "undefined" && document.visibilityState === "hidden"
  );
}

export function defaultTickMs() {
  const reduced =
    typeof window !== "undefined" &&
    typeof window.matchMedia === "function" &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  return reduced ? REDUCED_MOTION_TICK_MS : TICK_MS;
}
