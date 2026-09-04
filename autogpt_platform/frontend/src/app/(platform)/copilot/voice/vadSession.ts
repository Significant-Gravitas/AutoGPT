/**
 * Silero VAD in the browser, wrapped so the rest of voice mode never imports
 * onnxruntime.
 */

const VAD_OPTIONS = {
  positiveSpeechThreshold: 0.6,
  negativeSpeechThreshold: 0.45,
  // How long a pause may last before the turn is taken as finished. 700 ms
  // cut people off mid-thought; the model's own latency dwarfs the extra
  // wait, so erring long is nearly free.
  redemptionMs: 1540,
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
