/**
 * How loud AutoPilot is at this instant, read off the element that is
 * playing it, so the meter moves with the words rather than pretending to.
 */

import { runningAudioContext } from "./audioContext";

/**
 * A media element can be routed into Web Audio exactly once, and from then
 * on ALL of its sound goes through that graph — so the source is cached and
 * always reconnected to the destination. Get this wrong and voice mode goes
 * silent, which is worse than having no meter.
 */
let source: MediaElementAudioSourceNode | null = null;
let analyser: AnalyserNode | null = null;
let samples: Float32Array<ArrayBuffer> | null = null;
let attachedTo: HTMLAudioElement | null = null;

/**
 * Idempotent, and a no-op unless the context is running: attaching to a
 * suspended context would mute playback outright.
 */
export function watchSpeech(audio: HTMLAudioElement) {
  if (attachedTo === audio) return;
  const context = runningAudioContext();
  if (!context) return;
  try {
    source = context.createMediaElementSource(audio);
    analyser = context.createAnalyser();
    analyser.fftSize = 256;
    samples = new Float32Array(new ArrayBuffer(analyser.fftSize * 4));
    // Destination as well as analyser — an analyser alone is a dead end and
    // the audio would never reach the speakers.
    source.connect(analyser).connect(context.destination);
    attachedTo = audio;
  } catch {
    // Already routed by an earlier call, or Web Audio is unavailable. The
    // meter falls back to its animation; playback is untouched either way.
    analyser = null;
  }
}

/** RMS of the audio playing right now, or `null` when nothing is watching. */
export function readSpeechLevel(): number | null {
  if (!analyser || !samples) return null;
  analyser.getFloatTimeDomainData(samples);
  let sum = 0;
  for (const sample of samples) sum += sample * sample;
  return Math.sqrt(sum / samples.length);
}

/** Test seam. The real graph outlives the tab, by design. */
export function resetSpeechWatch() {
  source = null;
  analyser = null;
  samples = null;
  attachedTo = null;
}
