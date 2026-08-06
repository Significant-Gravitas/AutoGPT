import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useAudioBars } from "../useAudioBars";

let activeBand = 0;
let nextFrame: FrameRequestCallback | undefined;
const disconnect = vi.fn();
const closeAudioContext = vi.fn(() => Promise.resolve());

class FakeAudioContext {
  state: AudioContextState = "running";
  sampleRate = 48_000;

  createAnalyser() {
    return {
      fftSize: 512,
      frequencyBinCount: 256,
      smoothingTimeConstant: 0,
      getByteFrequencyData(samples: Uint8Array) {
        samples.fill(0);
        const binWidth = 48_000 / 512;
        const ranges = [
          [80, 250],
          [250, 500],
          [500, 900],
          [900, 1600],
          [1600, 3000],
        ];
        const [minimum, maximum] = ranges[activeBand];
        const start = Math.max(1, Math.ceil(minimum / binWidth));
        const end = Math.min(samples.length, Math.ceil(maximum / binWidth));

        for (let index = start; index < end; index += 1) {
          samples[index] = 255;
        }
      },
    } as unknown as AnalyserNode;
  }

  createMediaStreamSource() {
    return {
      connect() {},
      disconnect,
    } as unknown as MediaStreamAudioSourceNode;
  }

  resume() {
    return Promise.resolve();
  }

  close() {
    return closeAudioContext();
  }
}

beforeEach(() => {
  activeBand = 0;
  nextFrame = undefined;
  disconnect.mockClear();
  closeAudioContext.mockClear();
  vi.stubGlobal("AudioContext", FakeAudioContext);
  vi.stubGlobal("requestAnimationFrame", (callback: FrameRequestCallback) => {
    nextFrame = callback;
    return 1;
  });
  vi.stubGlobal("cancelAnimationFrame", vi.fn());
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("useAudioBars", () => {
  it("updates each bar from its own voice-frequency band", () => {
    const { result, unmount } = renderHook(() =>
      useAudioBars({} as MediaStream),
    );

    act(() => nextFrame?.(performance.now() + 100));

    expect(result.current[0].get()).toBeGreaterThan(0);
    expect(result.current.slice(1).every((level) => level.get() === 0)).toBe(
      true,
    );

    activeBand = 4;
    act(() => nextFrame?.(performance.now() + 200));

    expect(result.current[4].get()).toBeGreaterThan(0);
    expect(result.current[1].get()).toBe(0);
    expect(result.current[2].get()).toBe(0);
    expect(result.current[3].get()).toBe(0);

    unmount();
    expect(disconnect).toHaveBeenCalledOnce();
    expect(closeAudioContext).toHaveBeenCalledOnce();
  });
});
