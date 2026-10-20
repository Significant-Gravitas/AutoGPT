/**
 * Silero VAD in the browser, wrapped so the rest of voice mode never imports
 * onnxruntime.
 */

import { reportMicLevel } from "./micLevel";

const VAD_OPTIONS = {
  positiveSpeechThreshold: 0.6,
  // Below this a frame no longer counts as speech. The library default is
  // 0.25; 0.45 read a quiet night-time voice as silence and cut the turn
  // off on the word after a pause.
  negativeSpeechThreshold: 0.4,
  // How long a pause may last before the turn is taken as finished. 700 ms
  // cut people off mid-thought; the model's own latency dwarfs the extra
  // wait, so erring long is nearly free.
  redemptionMs: 1500,
  // Keeps the first phoneme, which the threshold crossing would otherwise eat.
  preSpeechPadMs: 250,
  minSpeechMs: 400,
};

/** Copied out of node_modules by `scripts/copy-vad-assets.mjs`. */
const ASSET_PATH = "/vad/";

interface Callbacks {
  onSpeechStart: () => void;
  /** 16 kHz mono samples, already trimmed to the utterance. */
  onSpeechEnd: (wav: Blob) => void;
  /** Below `minSpeechMs` — a cough, a door, a keyboard. */
  onMisfire: () => void;
}

export interface VadSession {
  pause: () => void;
  resume: () => void;
  destroy: () => Promise<void>;
}

export async function startVadSession(
  callbacks: Callbacks,
): Promise<VadSession> {
  const { MicVAD, utils } = await import("@ricky0123/vad-web");

  const vad = await MicVAD.new({
    ...VAD_OPTIONS,
    model: "v5",
    baseAssetPath: ASSET_PATH,
    onnxWASMBasePath: ASSET_PATH,
    onSpeechStart: callbacks.onSpeechStart,
    // Frames stop arriving while the VAD is paused, which is what makes the
    // level indicator go flat exactly when the mic is shut.
    onFrameProcessed: (_probabilities, frame) => reportMicLevel(rms(frame)),
    onVADMisfire: callbacks.onMisfire,
    onSpeechEnd: (audio) => {
      const wav = utils.encodeWAV(audio, 1, 16000, 1, 16);
      callbacks.onSpeechEnd(new Blob([wav], { type: "audio/wav" }));
    },
  });

  await vad.start();

  return {
    pause: () => void vad.pause(),
    resume: () => void vad.start(),
    destroy: () => vad.destroy(),
  };
}

function rms(frame: Float32Array): number {
  let sum = 0;
  for (const sample of frame) sum += sample * sample;
  return Math.sqrt(sum / frame.length);
}
