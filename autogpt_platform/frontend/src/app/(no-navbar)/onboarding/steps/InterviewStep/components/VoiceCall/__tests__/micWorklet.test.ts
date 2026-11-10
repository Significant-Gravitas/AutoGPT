import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";

// The audio thread hands a processor 128 samples at a time.
const QUANTUM = 128;
const FRAME = 640;

interface Capture {
  filled: number;
  process: (inputs: Float32Array[][]) => boolean;
}

type CaptureClass = new (options: unknown) => Capture;

// The capture runs in the audio thread's own scope; its globals are handed
// in here instead, and every frame it posts is kept.
function newCapture(sampleRate: number) {
  const frames: Int16Array[] = [];
  class FakeProcessor {
    port = {
      postMessage: (buffer: ArrayBuffer) => frames.push(new Int16Array(buffer)),
    };
  }
  const source = readFileSync(
    resolve(process.cwd(), "public/voice/mic-worklet.js"),
    "utf8",
  );
  let Processor: CaptureClass | null = null;
  const load = new Function(
    "AudioWorkletProcessor",
    "registerProcessor",
    "sampleRate",
    source,
  );
  load(
    FakeProcessor,
    (_name: string, processor: CaptureClass) => {
      Processor = processor;
    },
    sampleRate,
  );
  if (!Processor) throw new Error("The capture registered no processor");
  const capture = new (Processor as CaptureClass)({
    processorOptions: { targetRate: 16_000, frameMs: 40 },
  });
  return { capture, frames };
}

function feed(capture: Capture, samples: Float32Array) {
  for (let at = 0; at < samples.length; at += QUANTUM) {
    capture.process([[samples.subarray(at, at + QUANTUM)]]);
  }
}

function constant(length: number, value: number) {
  return new Float32Array(length).fill(value);
}

describe("the call's microphone capture", () => {
  // Two samples past 40 ms, so the last output's place, which 44.1 kHz puts
  // on a fraction, is behind the input rather than on its edge.
  it.each([
    [48_000, 1_920],
    [44_100, 1_764],
  ])(
    "turns 40 ms at %s Hz into one frame of 640 samples at 16 kHz",
    (rate, samplesIn40ms) => {
      const { capture, frames } = newCapture(rate);

      feed(capture, constant(samplesIn40ms + 2, 0.25));

      expect(frames).toHaveLength(1);
      expect(frames[0]).toHaveLength(FRAME);
      expect(capture.filled).toBe(0);
    },
  );

  it.each([[48_000], [44_100]])(
    "keeps to 16 kHz over a minute at %s Hz, the fraction of a sample carried from frame to frame",
    (rate) => {
      const { capture, frames } = newCapture(rate);

      feed(capture, constant(rate * 60 + 2, 0.25));

      // A minute at 16 kHz is 960,000 samples: 1,500 frames, none partial.
      expect(frames).toHaveLength(1_500);
      expect(capture.filled).toBe(0);
    },
  );

  it.each([
    [1, 32_767],
    [0.5, 16_383],
    [0, 0],
    [-0.5, -16_384],
    [-1, -32_768],
    [1.7, 32_767],
    [-2, -32_768],
  ])(
    "writes %s as %s in PCM16, clipping what is out of range",
    (value, expected) => {
      const { capture, frames } = newCapture(48_000);

      feed(capture, constant(1_920, value));

      expect(frames).toHaveLength(1);
      expect(new Set(frames[0])).toEqual(new Set([expected]));
    },
  );

  it("averages the samples each output covers", () => {
    const { capture, frames } = newCapture(48_000);
    const samples = new Float32Array(1_920);
    for (let i = 0; i < samples.length; i++) {
      samples[i] = [0.3, 0.6, 0.9][i % 3];
    }

    feed(capture, samples);

    // Each output covers three: their mean is 0.6, and 0.6 × 32,767 is
    // 19,660 once the fraction goes.
    expect(new Set(frames[0])).toEqual(new Set([19_660]));
  });
});
