import { describe, expect, it, vi } from "vitest";

const micVad = {
  start: vi.fn(async () => undefined),
  pause: vi.fn(),
  destroy: vi.fn(async () => undefined),
};
let captured: Record<string, unknown> = {};

vi.mock("@ricky0123/vad-web", () => ({
  MicVAD: {
    new: vi.fn(async (options: Record<string, unknown>) => {
      captured = options;
      return micVad;
    }),
  },
  utils: { encodeWAV: vi.fn(() => new ArrayBuffer(8)) },
}));

import { startVadSession } from "../vadSession";

const callbacks = {
  onSpeechStart: vi.fn(),
  onSpeechEnd: vi.fn(),
  onMisfire: vi.fn(),
};

describe("startVadSession", () => {
  it("uses the measured thresholds", async () => {
    await startVadSession(callbacks);

    expect(captured).toMatchObject({
      model: "v5",
      positiveSpeechThreshold: 0.6,
      // 500 ms split single utterances into several segments in the spike.
      redemptionMs: 700,
      preSpeechPadMs: 250,
      minSpeechMs: 400,
    });
  });

  it("loads its model from our own origin, not a CDN", async () => {
    await startVadSession(callbacks);

    expect(captured.baseAssetPath).toBe("/vad/");
    expect(captured.onnxWASMBasePath).toBe("/vad/");
  });

  it("hands the utterance over as a WAV blob the transcribe route accepts", async () => {
    await startVadSession(callbacks);

    const onSpeechEnd = captured.onSpeechEnd as (audio: Float32Array) => void;
    onSpeechEnd(new Float32Array(16000));

    const [wav] = callbacks.onSpeechEnd.mock.calls.at(-1)!;
    expect(wav.type).toBe("audio/wav");
  });

  it("starts listening and can be paused and torn down", async () => {
    const session = await startVadSession(callbacks);
    expect(micVad.start).toHaveBeenCalled();

    session.pause();
    expect(micVad.pause).toHaveBeenCalled();

    await session.destroy();
    expect(micVad.destroy).toHaveBeenCalled();
  });
});
