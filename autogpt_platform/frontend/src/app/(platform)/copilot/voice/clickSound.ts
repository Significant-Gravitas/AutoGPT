/**
 * The "heard you" cue, played the instant the user stops speaking.
 *
 * It replaces a spoken canned phrase: synthesised speech costs a round trip
 * and a credit, cannot be chosen well before the transcript exists, and
 * sounds canned by the third time. A tone is instant and free.
 *
 * Synthesised rather than loaded from a file so it costs no request and
 * cannot be late — the whole point is that it lands with the silence.
 */

import { runningAudioContext } from "./audioContext";

const ATTACK_S = 0.004;
const DECAY_S = 0.09;
const PEAK_GAIN = 0.18;
const START_HZ = 660;
const END_HZ = 520;

/** A soft downward pop. No-op if the browser never granted us a context. */
export function playClickSound() {
  const context = runningAudioContext();
  if (!context) return;
  try {
    const now = context.currentTime;
    const oscillator = context.createOscillator();
    const gain = context.createGain();

    oscillator.type = "sine";
    oscillator.frequency.setValueAtTime(START_HZ, now);
    oscillator.frequency.exponentialRampToValueAtTime(END_HZ, now + DECAY_S);

    // Ramping to an audible floor rather than 0: exponential ramps reject
    // zero, and stopping on a non-zero gain is what makes a click audible
    // as a click.
    gain.gain.setValueAtTime(0, now);
    gain.gain.linearRampToValueAtTime(PEAK_GAIN, now + ATTACK_S);
    gain.gain.exponentialRampToValueAtTime(0.0001, now + DECAY_S);

    oscillator.connect(gain).connect(context.destination);
    oscillator.start(now);
    oscillator.stop(now + DECAY_S);
  } catch {
    // A tone is never worth interrupting a conversation for.
  }
}
