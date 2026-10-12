/**
 * Turns a raw RMS into a bar height, adapting to whatever the source
 * actually gives.
 *
 * Fixed thresholds cannot work here: microphone gain varies by an order of
 * magnitude between machines, so any absolute "this is loud" either gates a
 * quiet mic to a flat line or pins a hot one at full scale. This tracks the
 * room and the recent peak instead and draws the distance between them.
 */

/** Recent loudest, decayed — about a 4.5 s half-life at a 67 ms tick. */
const CEILING_DECAY = 0.99;
/** The room drifts up slowly and drops to a new quiet immediately. */
const NOISE_RISE = 0.001;
/** Never stretch pure silence across the whole strip. */
const MIN_SPAN = 0.01;
/** Ignore this much of the span above the room, so noise reads as flat. */
const DEAD_ZONE = 0.08;
/** Gentler than a square root near zero, which made room tone look live. */
const CURVE = 0.7;

export function createLevelScale() {
  let ceiling = 0;
  let noise = Number.POSITIVE_INFINITY;

  return function scale(level: number): number {
    // A shut mic reports nothing at all. Letting that define the room would
    // make the next word arrive against a floor of silence.
    if (level > 0) {
      noise = level < noise ? level : noise + (level - noise) * NOISE_RISE;
    }
    ceiling = Math.max(level, ceiling * CEILING_DECAY);

    const span = Math.max(ceiling - noise, MIN_SPAN);
    const excess = level - noise - span * DEAD_ZONE;
    if (excess <= 0) return 0;
    return Math.min(1, (excess / span) ** CURVE);
  };
}
