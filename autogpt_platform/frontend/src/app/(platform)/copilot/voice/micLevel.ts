/**
 * The mic's loudness, handed from the VAD to the indicator that draws it.
 *
 * A module variable rather than React state: frames arrive ~31 times a
 * second, and re-rendering the chat that often to move a few pixels is not
 * worth it. The indicator samples this on its own, slower clock.
 */

let peak = 0;

/** Root-mean-square of one VAD frame, roughly 0–1. */
export function reportMicLevel(level: number) {
  peak = Math.max(peak, level);
}

/**
 * The loudest frame since the last call, then back to zero. Peak rather
 * than latest so a syllable falling between two samples still shows up —
 * and so a closed mic reads as silence within one interval.
 */
export function takeMicLevel(): number {
  const level = peak;
  peak = 0;
  return level;
}
